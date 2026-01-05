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
    pypy_token: str

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
    exact_search: bool = False
    exact_search_match: bool = False

    # Context parameters
    product_name: str | None = None
    product_price: str | None = None
    product_description: str | None = None
    product_images: List[str] | None = None
    probability: float | None = None
    html: str | None = None
    html_clean: str | None = None

    # Processor parameters are set dynamically
    classifications: Dict[str, int] = Field(default_factory=dict)
    insights: Dict[str, Dict[str, str]] | None = Field(default=None)

    # Temporary data for processor-specific intermediate data
    tmp: Dict[str, Any] = Field(default_factory=dict)

    # Usage parameters
    usage: Dict[str, Dict[str, int]] = Field(default_factory=dict)

    # Filtering parameters
    filtered: bool = False
    filtered_at_stage: str | None = None


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
    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

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
