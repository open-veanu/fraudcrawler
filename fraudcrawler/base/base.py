import json
import logging
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings
from urllib.parse import urlparse
import re
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from bs4.element import Tag
import httpx

from fraudcrawler.settings import (
    GOOGLE_LANGUAGES_FILENAME,
    GOOGLE_LOCATIONS_FILENAME,
)
from fraudcrawler.settings import (
    DEFAULT_HTTPX_TIMEOUT,
    DEFAULT_HTTPX_LIMITS,
    DEFAULT_HTTPX_REDIRECTS,
)
from fraudcrawler.settings import TOPPREISE_SEARCH_PATHS, TOPPREISE_COMPARISON_PATHS

logger = logging.getLogger(__name__)

# Load google locations and languages
with open(GOOGLE_LOCATIONS_FILENAME, "r") as gfile:
    _locs = json.load(gfile)
_LOCATION_CODES = {loc["name"]: loc["country_code"].lower() for loc in _locs}
with open(GOOGLE_LANGUAGES_FILENAME, "r") as gfile:
    _langs = json.load(gfile)
_LANGUAGE_CODES = {lang["language_name"]: lang["language_code"] for lang in _langs}


# Base classes
class Setup(BaseSettings):
    """Class for loading environment variables."""

    # Crawler ENV variables
    serpapi_key: str
    dataforseo_user: str
    dataforseo_pwd: str
    zyteapi_key: str
    openaiapi_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Host(BaseModel):
    """Model for host details (e.g. `Host(name="Galaxus", domains="galaxus.ch, digitec.ch")`)."""

    name: str
    domains: str | List[str]

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        """Make it lowercase and strip 'www.' and 'https?://' prefixes from the domain."""
        domain = domain.strip().lower()
        return re.sub(r"^(https?://)?(www\.)?", "", domain)

    @field_validator("domains", mode="before")
    def normalize_domains(cls, val):
        if isinstance(val, str):
            val = val.split(",")
        return [cls._normalize_domain(dom.strip()) for dom in val]


class ClassificationResult(BaseModel):
    """Model for classification results."""

    result: int
    input_tokens: int
    output_tokens: int


class Location(BaseModel):
    """Model for location details (e.g. `Location(name="Switzerland", code="ch")`)."""

    name: str
    code: str = ""

    @model_validator(mode="before")
    def set_code(cls, values):
        """Set the location code if not provided and make it lower case."""
        name = values.get("name")
        code = values.get("code")
        if code is None or not len(code):
            code = _LOCATION_CODES.get(name)
            if code is None:
                raise ValueError(f'Location code not found for location name="{name}"')
        code = code.lower()
        return {"name": name, "code": code}


class Language(BaseModel):
    """Model for language details (e.g. `Language(name="German", code="de")`)."""

    name: str
    code: str = ""

    @model_validator(mode="before")
    def set_code(cls, values):
        """Set the language code if not provided and make it lower case."""
        name = values.get("name")
        code = values.get("code")
        if code is None or not len(code):
            code = _LANGUAGE_CODES.get(name)
            if code is None:
                raise ValueError(f'Language code not found for language name="{name}"')
        code = code.lower()
        return {"name": name, "code": code}


class Enrichment(BaseModel):
    """Model for enriching initial search_term with alternative ones."""

    additional_terms: int
    additional_urls_per_term: int


class Deepness(BaseModel):
    """Model for search depth."""

    num_results: int
    enrichment: Enrichment | None = None


class ProductItem(BaseModel):
    """Model representing a product item."""

    # Search parameters
    search_term: str
    search_term_type: str
    url: str
    url_resolved: str
    search_engine_name: str
    domain: str

    # Context parameters
    product_name: str | None = None
    product_price: str | None = None
    product_description: str | None = None
    product_images: List[str] | None = None
    probability: float | None = None
    html: str | None = None
    html_clean: str | None = None

    # Processor parameters are set dynamic so we must allow extra fields
    classifications: Dict[str, int] = Field(default_factory=dict)

    # Usage parameters
    usage: Dict[str, Dict[str, int]] = Field(default_factory=dict)

    # Filtering parameters
    filtered: bool = False
    filtered_at_stage: str | None = None


class Prompt(BaseModel):
    """Model for prompts."""

    name: str
    system_prompt: str
    product_item_fields: List[str]
    allowed_classes: List[int]

    @field_validator("allowed_classes", mode="before")
    def check_for_positive_value(cls, val):
        """Check if all values are positive."""
        if not all(isinstance(i, int) and i >= 0 for i in val):
            raise ValueError("all values in allowed_classes must be positive integers.")
        return val

    @field_validator("product_item_fields", mode="before")
    def validate_product_item_fields(cls, val):
        """Ensure all product_item_fields are valid ProductItem attributes."""
        valid_fields = set(ProductItem.model_fields.keys())
        for field in val:
            if field not in valid_fields:
                raise ValueError(
                    f"Invalid product_item_field: '{field}'. Must be one of: {sorted(valid_fields)}"
                )
        return val


class HttpxAsyncClient(httpx.AsyncClient):
    """Httpx async client that can be used to retain the default settings."""

    def __init__(
        self,
        timeout: httpx.Timeout | Dict[str, Any] = DEFAULT_HTTPX_TIMEOUT,
        limits: httpx.Limits | Dict[str, Any] = DEFAULT_HTTPX_LIMITS,
        follow_redirects: bool = DEFAULT_HTTPX_REDIRECTS,
        **kwargs: Any,
    ) -> None:
        if isinstance(timeout, dict):
            timeout = httpx.Timeout(**timeout)
        if isinstance(limits, dict):
            limits = httpx.Limits(**limits)

        kwargs.setdefault("timeout", timeout)
        kwargs.setdefault("limits", limits)
        kwargs.setdefault("follow_redirects", follow_redirects)
        super().__init__(**kwargs)


class DomainUtils:
    """Utility class for domain extraction and normalization.

    Handles domain parsing from URLs, removes common prefixes (www, http/https),
    and provides consistent domain formatting for search and scraping operations.
    """

    _hostname_pattern = r"^(?:https?:\/\/)?([^\/:?#]+)"

    def _get_domain(self, url: str) -> str:
        """Extracts the second-level domain together with the top-level domain (e.g. `google.com`).

        Args:
            url: The URL to be processed.
        """
        # Add scheme; urlparse requires it
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        # Get the hostname
        hostname = urlparse(url).hostname
        if hostname is None and (match := re.search(self._hostname_pattern, url)):
            hostname = match.group(1)
        if hostname is None:
            logger.warning(
                f'Failed to extract domain from url="{url}"; full url is returned'
            )
            return url.lower()

        # Remove www. prefix
        if hostname and hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname.lower()


class ToppreiseUtils:
    """Utility class for Toppreise specific URL extraction."""

    _endpoint = "https://www.toppreise.ch/"
    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    @classmethod
    def _get_search_endpoint(cls, language: Language) -> str:
        """Get the search endpoint based on the language."""
        search_path = TOPPREISE_SEARCH_PATHS.get(
            language.code, TOPPREISE_SEARCH_PATHS["default"]
        )
        return f"{cls._endpoint}{search_path}"

    @staticmethod
    def _extract_links(
        element: Tag, ext_products: bool = True, comp_products: bool = True
    ) -> List[str]:
        """Extracts all relevant product URLs from a BeautifulSoup object of a Toppreise page.

        Note:
            Depending on the arguments, it extracts:
                - product comparison URLs (i.e. https://www.toppreise.ch/preisvergleich/...)
                - external product URLs (i.e. https://www.example.com/ext_...).

        Args:
            tag: BeautifulSoup Tag object containing the HTML to parse.
            ext_products: Whether to extract external product URLs.
            comp_products: Whether to extract product comparison URLs.
        """
        # Find all links in the page
        links = element.find_all("a", href=True)

        # Filter links to only include external product links
        hrefs = [
            href
            for link in links
            if (
                hasattr(link, "get")  # Ensure we have a Tag object with href attribute
                and (href := link.get("href"))  # Ensure href is not None
                and not href.startswith("javascript:")  # Skip javascript links
                and isinstance(href, str)  # Ensure href is a string
                # Make sure the link is either an external product link (href contains 'ext_')
                # or is a search result link (href contains 'preisvergleich', 'comparison-prix', or 'price-comparison')
                and (
                    ("ext_" in href and ext_products)
                    or (
                        any(pth in href for pth in TOPPREISE_COMPARISON_PATHS)
                        and comp_products
                    )
                )
            )
        ]

        # Make relative URLs absolute
        urls = []
        for href in hrefs:
            if href.startswith("/"):
                href = f"https://www.toppreise.ch{href}"
            elif not href.startswith("http"):
                href = f"https://www.toppreise.ch/{href}"
            urls.append(href)

        # Return deduplicated urls
        urls = list(set(urls))
        return urls

    def _extract_product_urls_from_search_page(self, content: bytes) -> List[str]:
        """Extracts product urls from a Toppreise search page (i.e. https://www.toppreise.ch/produktsuche)."""

        # Parse the HTML
        soup = BeautifulSoup(content, "html.parser")
        main = soup.find("div", id="Page_Browsing")
        if not isinstance(main, Tag):
            logger.warning("No main content found in Toppreise search page.")
            return []

        # Extract links (external product links and comparison links)
        urls = self._extract_links(element=main)

        logger.debug(f"Found {len(urls)} product URLs from Toppreise search results.")
        return urls

    def _extract_product_urls_from_comparison_page(self, content: bytes) -> List[str]:
        """Extracts product urls from a Toppreise product comparison page (i.e. https://www.toppreise.ch/preisvergleich/...)."""

        # Parse the HTML
        soup = BeautifulSoup(content, "html.parser")

        # Extract links (external product links only)
        urls = self._extract_links(element=soup, comp_products=False)

        logger.debug(
            f"Found {len(urls)} external product URLs from Toppreise comparison page."
        )
        return urls
