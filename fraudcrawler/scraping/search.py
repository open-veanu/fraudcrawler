from abc import ABC, abstractmethod
from enum import Enum
import logging
import re
from pydantic import BaseModel, Field
from typing import Awaitable, Callable, Dict, List, Optional, Sequence
import unicodedata
from urllib.parse import (
    ParseResult,
    parse_qsl,
    quote_plus,
    urlencode,
    urlparse,
    urlunparse,
)

from bs4 import BeautifulSoup
from bs4.element import Tag
import httpx
from tenacity import RetryCallState

from fraudcrawler.settings import (
    SEARCH_DEFAULT_COUNTRY_CODES,
    TOPPREISE_SEARCH_PATHS,
    TOPPREISE_COMPARISON_PATHS,
    REDIS_USE_CACHE,
)
from fraudcrawler.base.base import (
    Host,
    Language,
    Location,
    DomainUtils,
    WebsiteSourceMetadata,
)
from fraudcrawler.base.retry import get_async_retry
from fraudcrawler.cache.cacher import RedisCacher
from fraudcrawler.scraping.url import filter_tracking_query_entries
from fraudcrawler.scraping.zyte import (
    SavedSearchRenderedProductListItem,
    ZyteAPI,
)
from fraudcrawler.scraping.saved_search_models import (
    WebsiteSource,
    WebsiteSourceSearchableUrl,
    WebsiteSourceUrlTemplate,
)

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Model for a single search result."""

    url: str
    domain: str
    search_engine_name: str
    filtered: bool = False
    filtered_at_stage: str | None = None
    website_source: WebsiteSourceMetadata | None = None
    candidate_title: str | None = None
    candidate_price: str | None = None
    candidate_images: List[str] | None = None


class SavedSearchCandidate(BaseModel):
    url: str
    title: str
    image_urls: List[str] = Field(default_factory=list, alias="imageUrls")
    price: str | None = None


class SavedSearchUrlDiagnostic(BaseModel):
    url: str
    resolved_url: str | None = Field(default=None, alias="resolvedUrl")
    render_http_status: int | None = Field(default=None, alias="renderHttpStatus")
    render_error: str | None = Field(default=None, alias="renderError")
    fetched: int = 0
    parsed: int = 0
    deduped: int = 0
    error: str | None = None


class SavedSearchIngestResult(BaseModel):
    source_name: str = Field(alias="sourceName")
    source_urls: List[str] = Field(default_factory=list, alias="sourceUrls")
    candidates: List[SavedSearchCandidate] = Field(default_factory=list)
    samples: List[SavedSearchCandidate] = Field(default_factory=list)
    fetched: int = 0
    parsed: int = 0
    deduped: int = 0
    url_diagnostics: List[SavedSearchUrlDiagnostic] = Field(
        default_factory=list, alias="urlDiagnostics"
    )


class SearchEngineName(Enum):
    """Enum for search engine names."""

    GOOGLE = "google"
    GOOGLE_SHOPPING = "google_shopping"
    TOPPREISE = "toppreise"
    WEBSITE_SOURCE = "website_source"


class SearchEngine(ABC, DomainUtils):
    """Abstract base class for search engines."""

    _hostname_pattern = r"^(?:https?:\/\/)?([^\/:?#]+)"

    def __init__(self, http_client: httpx.AsyncClient):
        """Initializes the SearchEngine with the given HTTP client.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
        """
        self._http_client = http_client
        super().__init__()

    @property
    @abstractmethod
    def _search_engine_name(self) -> str:
        """The name of the search engine."""
        pass

    @abstractmethod
    async def search(self, *args, **kwargs) -> List[SearchResult]:
        """Apply the search with the given parameters and return results."""
        pass

    def _create_search_result(self, url: str) -> SearchResult:
        """From a given url it creates the class:`SearchResult` instance."""
        # Get marketplace name
        domain = self._get_domain(url=url)

        # Create and return the SearchResult object
        result = SearchResult(
            url=url,
            domain=domain,
            search_engine_name=self._search_engine_name,
        )
        return result

    @classmethod
    def _log_before(
        cls, url: str, params: dict | None, retry_state: RetryCallState | None
    ) -> None:
        """Context aware logging before HTTP request is made."""
        if retry_state:
            logger.debug(
                f'Performing HTTP request in {cls.__name__} to url="{url}" '
                f"with params={params} (attempt {retry_state.attempt_number})."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before.")

    @classmethod
    def _log_before_sleep(
        cls, url: str, params: dict | None, retry_state: RetryCallState | None
    ) -> None:
        """Context aware logging before sleeping after a failed HTTP request."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f"Attempt {retry_state.attempt_number} of {cls.__name__} HTTP request "
                f'to url="{url}" with params="{params}" '
                f"failed with error: {retry_state.outcome.exception()}. "
                f"Retrying in {retry_state.upcoming_sleep:.0f} seconds."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before_sleep.")

    async def http_client_get(
        self, url: str, params: dict | None = None, headers: dict | None = None
    ) -> httpx.Response:
        """Performs a GET request with retries.

        Args:
            retry: The retry strategy to use.
            url: The URL to request.
            params: Query parameters for the request.
            headers: HTTP headers to use for the request.
        """
        # Perform the request and retry if necessary. There is some context aware logging:
        #  - `before`: before the request is made (and before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            url=url, params=params, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            url=url, params=params, retry_state=retry_state
        )

        async for attempt in retry:
            with attempt:
                response = await self._http_client.get(
                    url=url,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                return response

        # In case of not entering the for loop (for some strange reason)
        raise RuntimeError("Retry exhausted without success")


class SerpAPI(SearchEngine):
    """Base class for SerpAPI search engines."""

    _endpoint = "https://serpapi.com/search"

    def __init__(self, http_client: httpx.AsyncClient, api_key: str):
        """Initializes the SerpAPI client with the given API key.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            api_key: The API key for SerpAPI.
        """
        super().__init__(http_client=http_client)
        self._api_key = api_key

    @property
    @abstractmethod
    def _engine(self) -> str:
        """The search engine name used in the SerpAPI request."""
        pass

    @staticmethod
    @abstractmethod
    def _extract_search_results_urls(data: dict) -> List[str]:
        """Extracts search results urls from the response.

        Args:
            data: The json from the SerpAPI search response.
        """
        pass

    @staticmethod
    def _get_search_string(search_term: str, marketplaces: List[Host] | None) -> str:
        """Constructs the search string with site: parameters for marketplaces."""
        search_string = search_term
        if marketplaces:
            sites = [dom for host in marketplaces for dom in host.domains]
            search_string += " site:" + " OR site:".join(s for s in sites)
        return search_string

    @staticmethod
    def _get_google_domain(location: Location) -> str:
        """Gets the Google domain for the given location if they do not use the default pattern google.tld"""
        if location.name == "Brazil":
            return "google.com.br"
        elif location.name == "United Kingdom":
            return "google.co.uk"
        elif location.name == "Argentina":
            return "google.com.ar"
        return f"google.{location.code}"

    async def _search(
        self,
        search_string: str,
        language: Language,
        location: Location,
        num_results: int,
    ) -> List[str]:
        """Performs a search using SerpAPI and returns the URLs of the results.

        Args:
            search_string: The search string to use (with potentially added site: parameters).
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.

        The SerpAPI parameters are:
            engine: The search engine to use ('google', 'google_shopping' etc.).
            q: The search string (with potentially added site: parameters).
            google_domain: The Google domain to use for the search (e.g. google.[com]).
            location_[requested|used]: The location to use for the search.
            tbs: The to-be-searched  parameters (e.g. 'ctr:CH').
            cr: The country code to limit the search to (e.g. 'countryCH').
            gl: The country code to use for the search.
            hl: The language code to use for the search.
            api_key: The API key to use for the search.

        Pagination:
            The 'num' parameter is not reliably supported by SerpAPI for google
            and google_shopping engines. Instead, we paginate using
            'serpapi_pagination.next_link' until num_results is reached or there
            are no more pages.
        """
        engine = self._engine

        # Log the search parameters
        logger.debug(
            f'Performing SerpAPI search with engine="{engine}", '
            f'q="{search_string}", '
            f'location="{location.name}", '
            f'language="{language.code}", '
            f"num_results={num_results}."
        )

        # Get Google domain and country code
        google_domain = self._get_google_domain(location)
        country_code = location.code

        params: Dict[str, str | int] = {
            "engine": engine,
            "q": search_string,
            "google_domain": google_domain,
            "location_requested": location.name,
            "location_used": location.name,
            "tbs": f"ctr:{country_code.upper()}",
            "cr": f"country{country_code.upper()}",
            "gl": country_code,
            "hl": language.code,
            "api_key": self._api_key,
        }
        logger.debug(f"SerpAPI search with params: {params}")

        # Extract urls for the first page
        response = await self.http_client_get(
            url=self._endpoint, params=params
        )
        data = response.json()
        urls = self._extract_search_results_urls(data=data)
        if len(urls) >= num_results:
            return urls

        # Paginate through SerpAPI results until num_results is reached or no more pages
        next_url: str | None = data.get("serpapi_pagination", {}).get("next_link")
        page = 2
        while len(urls) < num_results and next_url:
            try:
                response = await self.http_client_get(
                    url=f"{next_url}&api_key={self._api_key}"
                )
            except Exception:
                logger.warning(
                    f"SerpAPI pagination request failed on page {page} for q=\"{search_string}\" "
                    f"and engine=\"{engine}\". Returning {len(urls)} URLs collected so far."
                )
                break

            data = response.json()
            page_urls = self._extract_search_results_urls(data=data)

            if not page_urls:
                break

            urls.extend(page_urls)
            logger.debug(
                f"SerpAPI page {page}: found {len(page_urls)} URLs "
                f"({len(urls)} total) for q=\"{search_string}\" and engine=\"{engine}\"."
            )

            # Update next_page and page
            next_url = data.get("serpapi_pagination", {}).get("next_link")
            page += 1

        logger.debug(
            f'Found total of {len(urls)} URLs from SerpAPI search for q="{search_string}" and engine="{engine}".'
        )
        return urls


class SerpAPIGoogle(SerpAPI):
    """Search engine for Google in SerpAPI."""

    def __init__(self, http_client: httpx.AsyncClient, api_key: str):
        """Initializes the SerpAPIGoogle client with the given API key.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            api_key: The API key for SerpAPI.
        """
        super().__init__(http_client=http_client, api_key=api_key)

    @property
    def _search_engine_name(self) -> str:
        """The name of the search engine."""
        return SearchEngineName.GOOGLE.value

    @property
    def _engine(self) -> str:
        """The search engine name used in the SerpAPI request."""
        return "google"

    @staticmethod
    def _extract_search_results_urls(data: dict) -> List[str]:
        """Extracts search results urls from the response data.

        Args:
            data: The json data from the SerpApi search response.
        """
        results = data.get("organic_results")
        if results is not None:
            return [url for res in results if (url := res.get("link"))]
        return []

    async def search(
        self,
        search_term: str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
    ) -> List[SearchResult]:
        """Performs a google search using SerpApi and returns SearchResults.

        Args:
            search_term: The search term to use for the query.
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.
            marketplaces: The marketplaces to include in the search.
        """
        # Construct the search string
        search_string = self._get_search_string(
            search_term=search_term,
            marketplaces=marketplaces,
        )

        # Perform the search
        urls = await self._search(
            search_string=search_string,
            language=language,
            location=location,
            num_results=num_results,
        )
        urls = urls[:num_results]

        # Create and return SearchResult objects from the URLs
        results = [self._create_search_result(url=url) for url in urls]
        logger.debug(
            f'Produced {len(results)} results from SerpAPI with engine="{self._engine}" and q="{search_string}".'
        )
        return results


class SerpAPIGoogleShopping(SerpAPI):
    """Search engine for Google Shopping in SerpAPI."""

    def __init__(self, http_client: httpx.AsyncClient, api_key: str):
        """Initializes the SerpAPIGoogleShopping client with the given API key.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            api_key: The API key for SerpAPI.
        """
        super().__init__(http_client=http_client, api_key=api_key)

    @property
    def _search_engine_name(self) -> str:
        """The name of the search engine."""
        return SearchEngineName.GOOGLE_SHOPPING.value

    @property
    def _engine(self) -> str:
        """The search engine name used in the SerpAPI request."""
        return "google_shopping"

    @staticmethod
    def _extract_search_results_urls(data: dict) -> List[str]:
        """Extracts search results urls from the response data.

        Args:
            data: The json data from the SerpApi search response.
        """
        results = data.get("shopping_results")
        if results is not None:
            # return [url for res in results if (url := res.get("product_link"))]   # c.f. https://github.com/serpapi/public-roadmap/issues/3045
            return [
                url
                for res in results
                if (url := res.get("serpapi_immersive_product_api"))
            ]
        return []

    @staticmethod
    def _extract_product_urls_from_immersive_product_api(data: dict) -> List[str]:
        """Extracts product urls from the serpapi immersive product API data."""
        if results := data.get("product_results"):
            stores = results.get("stores", [])
            urls = [url for sre in stores if (url := sre.get("link"))]
            return list(set(urls))
        return []

    async def search(
        self,
        search_term: str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
    ) -> List[SearchResult]:
        """Performs a google shopping search using SerpApi and returns SearchResults.

        Similar to Toppreise, this method extracts merchant URLs from Google Shopping product pages
        and creates multiple SearchResult objects for each merchant URL found.

        Args:
            search_term: The search term to use for the query.
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.
            marketplaces: The marketplaces to include in the search.
        """
        # Construct the search string
        search_string = self._get_search_string(
            search_term=search_term,
            marketplaces=marketplaces,
        )

        # Perform the search to get Google Shopping URLs
        urls = await self._search(
            search_string=search_string,
            language=language,
            location=location,
            num_results=num_results,
        )

        # !!! NOTE !!!: Google Shopping results do not properly support the 'num' parameter,
        # so we might get more results than requested. This is a known issue with SerpAPI
        # and Google Shopping searches (see https://github.com/serpapi/public-roadmap/issues/1858)
        urls = urls[:num_results]

        # Create SearchResult objects from merchant URLs (similar to Toppreise pattern)
        results = [self._create_search_result(url=url) for url in urls]
        logger.debug(
            f'Produced {len(results)} results from Google Shopping search with q="{search_string}".'
        )
        return results


class Toppreise(SearchEngine):
    """Search engine for toppreise.ch."""

    _endpoint = "https://www.toppreise.ch/"

    def __init__(
        self, http_client: httpx.AsyncClient, zyteapi_key: str, redis_use_cache: bool
    ):
        """Initializes the Toppreise client.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            zyteapi_key: ZyteAPI key for fallback when direct access fails.
            redis_use_cache: Whether to use cache (passed to internal ZyteAPI).
        """
        super().__init__(http_client=http_client)
        self._zyteapi = ZyteAPI(
            http_client=http_client,
            api_key=zyteapi_key,
            redis_use_cache=redis_use_cache,
        )

    async def http_client_get_with_fallback(self, url: str) -> bytes:
        """Performs a GET request with retries.

        If direct access fails (e.g. 403 Forbidden), it will attempt to unblock the URL
        content using Zyte proxy mode.

        Args:
            url: The URL to request.
        """
        # Try to access the URL directly
        try:
            response: httpx.Response = await self.http_client_get(
                url=url, headers=self._headers
            )
            content = response.content

        # If we get a 403 Error (can happen depending on IP/location of deployment),
        # we try to unblock the URL using Zyte proxy mode
        except httpx.HTTPStatusError as err_direct:
            if err_direct.response.status_code == 403:
                logger.warning(
                    f"Received 403 Forbidden for {url}, attempting to unblock with Zyte proxy"
                )
                try:
                    content = await self._zyteapi.unblock_url_content(url)
                except Exception as err_resolve:
                    msg = f'Error unblocking URL="{url}" with Zyte proxy: {err_resolve}'
                    logger.error(msg, exc_info=True)
                    raise httpx.HTTPError(msg) from err_resolve
            else:
                raise err_direct
        return content

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
                hasattr(link, "get")
                and (href := link.get("href"))  # type: ignore[attributeAccessIssue]
                and isinstance(
                    href, str
                )  # Ensure href is a string (excludes AttributeValueList)
                and not href.startswith("javascript:")  # Skip javascript links
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

    @property
    def _search_engine_name(self) -> str:
        """The name of the search engine."""
        return SearchEngineName.TOPPREISE.value

    async def _search(
        self, search_string: str, language: Language, num_results: int
    ) -> List[str]:
        """Performs a search on Toppreise and returns the URLs of the results.

        If direct access fails (e.g. 403 Forbidden), it will attempt to unblock the URL
        content using Zyte proxy mode.

        Args:
            search_string: The search string to use for the query.
            language: The language to use for the query.
            num_results: Max number of results to return.
        """
        # Build the search URL for Toppreise
        endpoint = self._get_search_endpoint(language=language)
        encoded_search = quote_plus(search_string)
        url = f"{endpoint}?q={encoded_search}"
        logger.debug(f"Toppreise search URL: {url}")

        # Perform the request with fallback if necessary
        content = await self.http_client_get_with_fallback(url=url)

        # Get external product urls from the content
        urls = self._extract_product_urls_from_search_page(content=content)
        urls = urls[:num_results]  # Limit to num_results if needed

        return urls

    async def search(
        self,
        search_term: str,
        language: Language,
        num_results: int,
    ) -> List[SearchResult]:
        """Performs a Toppreise search and returns SearchResults.

        Args:
            search_term: The search term to use for the query.
            language: The language to use for the search.
            num_results: Max number of results to return.
        """
        # Perform the search
        urls = await self._search(
            search_string=search_term,
            language=language,
            num_results=num_results,
        )

        # Create and return SearchResult objects from the URLs
        results = [self._create_search_result(url=url) for url in urls]
        logger.debug(
            f'Produced {len(results)} results from Toppreise search with q="{search_term}".'
        )
        return results


class WebsiteSearch(SearchEngine):
    """Search engine for website-source ingestion."""

    _saved_search_query_param_keys = ["q", "query", "keyword", "search"]
    _max_image_urls_per_candidate = 5

    @staticmethod
    def _build_website_source_engine_name(source_name: str) -> str:
        """Build a stable engine-like name from a website-source name."""
        ascii_name = (
            unicodedata.normalize("NFKD", str(source_name or ""))
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        normalized = re.sub(r"\s+", "_", ascii_name.strip().lower())
        normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        slug = normalized or "website_source"
        return f"{slug}_search_engine"

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        zyteapi_key: str,
        redis_use_cache: bool = REDIS_USE_CACHE,
    ):
        super().__init__(http_client=http_client)
        self._zyteapi = ZyteAPI(
            http_client=http_client,
            api_key=zyteapi_key,
            redis_use_cache=redis_use_cache,
        )

    @property
    def _search_engine_name(self) -> str:
        return SearchEngineName.WEBSITE_SOURCE.value

    @classmethod
    def _get_search_param_order(cls, param_key: str) -> int:
        try:
            return cls._saved_search_query_param_keys.index(param_key.lower())
        except ValueError:
            return 10_000

    @classmethod
    def _canonicalize_url(cls, raw_url: str) -> str:
        parsed = urlparse(raw_url.strip())
        host = parsed.hostname.lower() if parsed.hostname else ""
        netloc = host
        if parsed.port and not (
            (parsed.scheme == "https" and parsed.port == 443)
            or (parsed.scheme == "http" and parsed.port == 80)
        ):
            netloc = f"{host}:{parsed.port}"

        query_entries = [
            entry for entry in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        query_entries = filter_tracking_query_entries(query_entries)
        query_entries = sorted(
            query_entries,
            key=lambda item: (
                cls._get_search_param_order(item[0]),
                item[0].lower(),
                item[1],
            ),
        )
        query = urlencode(query_entries, quote_via=quote_plus)
        cleaned = ParseResult(
            scheme=parsed.scheme,
            netloc=netloc,
            path=parsed.path,
            params=parsed.params,
            query=query,
            fragment="",
        )
        return urlunparse(cleaned)

    @classmethod
    def _normalize_url(cls, base_url: str, href: str) -> Optional[str]:
        try:
            base = httpx.URL(base_url)
            joined = base.join(href)
        except (ValueError, httpx.InvalidURL):
            logger.debug(
                f"Failed to join URL base={base_url} href={href}", exc_info=True
            )
            return None
        try:
            return cls._canonicalize_url(str(joined))
        except (ValueError, httpx.InvalidURL):
            logger.debug(f"Failed to canonicalize URL {joined}", exc_info=True)
            return None

    @staticmethod
    def _interpolate_search_term(value: str, search_term: Optional[str]) -> str:
        if "{search_term}" not in value:
            return value
        encoded = quote_plus(search_term or "")
        return value.replace("{search_term}", encoded)

    def _build_searchable_url(
        self,
        template: WebsiteSourceUrlTemplate,
        searchable_url: WebsiteSourceSearchableUrl,
        search_term: Optional[str],
    ) -> Optional[str]:
        interpolated = self._interpolate_search_term(
            searchable_url.filter_url,
            search_term,
        )
        return self._normalize_url(base_url=template.base_url, href=interpolated)

    @classmethod
    def _extract_candidates_from_product_list(
        cls,
        *,
        items: List[SavedSearchRenderedProductListItem],
        source_url: str,
        max_items: int,
    ) -> List[SavedSearchCandidate]:
        candidates: List[SavedSearchCandidate] = []
        seen = set()
        for item in items:
            normalized = cls._normalize_url(source_url, item.url or "")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)

            image_urls: List[str] = []
            for raw in [item.main_image, *(item.images or [])]:
                token = str(raw or "").strip()
                if not token or token.startswith(
                    ("data:", "blob:", "javascript:", "#")
                ):
                    continue
                normalized_image = cls._normalize_url(source_url, token)
                if not normalized_image or not normalized_image.startswith(
                    ("http://", "https://")
                ):
                    continue
                if normalized_image in image_urls:
                    continue
                image_urls.append(normalized_image)
                if len(image_urls) >= cls._max_image_urls_per_candidate:
                    break

            price_value: str | None = None
            if isinstance(item.price, (int, float)) and item.price > 0:
                price_value = str(item.price)
            elif isinstance(item.price, str):
                cleaned_price = re.sub(r"\s+", " ", item.price).strip()
                if re.search(r"\d", cleaned_price):
                    price_value = cleaned_price[:128]

            title = (item.name or "").strip() or "Recovered from Zyte product list"

            candidates.append(
                SavedSearchCandidate(
                    url=normalized,
                    title=title,
                    imageUrls=image_urls,
                    price=price_value,
                )
            )
            if len(candidates) >= max_items:
                break
        return candidates

    @staticmethod
    def _apply_url_pattern_filters(
        candidates: Sequence[SavedSearchCandidate],
        include_substrings: Sequence[str],
        exclude_substrings: Sequence[str],
    ) -> List[SavedSearchCandidate]:
        include_tokens = [
            str(token).strip().lower()
            for token in include_substrings
            if str(token).strip()
        ]
        exclude_tokens = [
            str(token).strip().lower()
            for token in exclude_substrings
            if str(token).strip()
        ]
        if not include_tokens and not exclude_tokens:
            return list(candidates)
        filtered: List[SavedSearchCandidate] = []
        for candidate in candidates:
            lowered_url = candidate.url.lower()
            if include_tokens and not all(
                token in lowered_url for token in include_tokens
            ):
                continue
            if exclude_tokens and any(token in lowered_url for token in exclude_tokens):
                continue
            filtered.append(candidate)
        return filtered

    async def ingest_source(
        self,
        source: WebsiteSource,
        search_term: str | None = None,
        max_items: int = 250,
    ) -> SavedSearchIngestResult:
        combined: List[SavedSearchCandidate] = []
        seen: dict[str, int] = {}
        diagnostics: List[SavedSearchUrlDiagnostic] = []
        resolved_urls: List[str] = []
        remaining = max(0, max_items)

        for template in source.urls:
            if remaining <= 0:
                break
            for searchable_url in template.searchable_urls:
                if remaining <= 0:
                    break
                resolved: str | None = None
                try:
                    resolved = self._build_searchable_url(
                        template=template,
                        searchable_url=searchable_url,
                        search_term=search_term,
                    )
                    logger.info(f"Resolved URL: {resolved}")
                    if resolved is None:
                        raise ValueError(
                            f"Failed to resolve searchableUrl='{searchable_url.filter_url}' against baseUrl='{template.base_url}'."
                        )
                    resolved_urls.append(resolved)
                    if source.search_filter_config is None:
                        raise ValueError(
                            f"Website-source '{source.name}' requires searchFilterConfig.renderOptions."
                        )
                    render_options = source.search_filter_config.render_options
                    rendered = await self._zyteapi.fetch_rendered_page(
                        url=resolved,
                        javascript=render_options.javascript,
                        include_iframes=render_options.include_iframes,
                        actions=render_options.actions or [],
                        network_capture=render_options.network_capture or [],
                        request_headers=(
                            render_options.request_headers.model_dump(exclude_none=True)
                            if render_options.request_headers
                            else None
                        ),
                    )
                    filtered = self._apply_url_pattern_filters(
                        candidates=self._extract_candidates_from_product_list(
                            items=rendered.product_list_items,
                            source_url=resolved,
                            max_items=remaining,
                        ),
                        include_substrings=searchable_url.include_substrings,
                        exclude_substrings=searchable_url.exclude_substrings,
                    )
                    deduped_count = 0
                    for candidate in filtered:
                        if candidate.url in seen:
                            idx = seen[candidate.url]
                            existing = combined[idx]
                            merged_images = list(existing.image_urls)
                            for image in candidate.image_urls:
                                if image in merged_images:
                                    continue
                                merged_images.append(image)
                                if (
                                    len(merged_images)
                                    >= self._max_image_urls_per_candidate
                                ):
                                    break
                            combined[idx] = SavedSearchCandidate(
                                url=existing.url,
                                title=existing.title or candidate.title,
                                imageUrls=merged_images,
                                price=candidate.price or existing.price,
                            )
                            deduped_count += 1
                            continue
                        seen[candidate.url] = len(combined)
                        combined.append(candidate)
                        remaining = max(0, remaining - 1)
                        if remaining <= 0:
                            break

                    diagnostics.append(
                        SavedSearchUrlDiagnostic(
                            url=searchable_url.filter_url,
                            resolvedUrl=resolved,
                            renderHttpStatus=rendered.status_code,
                            renderError=None,
                            fetched=1
                            if rendered.status_code
                            and 200 <= rendered.status_code < 300
                            else 0,
                            parsed=len(filtered),
                            deduped=deduped_count,
                        )
                    )
                except Exception as err:
                    logger.warning(
                        "Website-source render failed | source=%s | filter_url=%s | resolved=%s | error=%s",
                        source.name,
                        searchable_url.filter_url,
                        resolved,
                        err,
                    )
                    diagnostics.append(
                        SavedSearchUrlDiagnostic(
                            url=searchable_url.filter_url,
                            resolvedUrl=resolved,
                            renderHttpStatus=None,
                            renderError=str(err),
                            fetched=0,
                            parsed=0,
                            deduped=0,
                            error=str(err),
                        )
                    )

        combined = combined[:max_items]
        return SavedSearchIngestResult(
            sourceName=source.name,
            sourceUrls=resolved_urls,
            candidates=combined,
            samples=combined[:10],
            fetched=sum(item.fetched for item in diagnostics),
            parsed=len(combined),
            deduped=sum(item.deduped for item in diagnostics),
            urlDiagnostics=diagnostics,
        )

    async def search(
        self,
        source: WebsiteSource,
        search_term: str | None = None,
        num_results: int = 250,
    ) -> List[SearchResult]:
        ingest_result = await self.ingest_source(
            source=source,
            search_term=search_term,
            max_items=num_results,
        )
        search_engine_name = self._build_website_source_engine_name(
            ingest_result.source_name
        )

        def _candidate_metadata(candidate_url: str) -> WebsiteSourceMetadata:
            diagnostics: List[SavedSearchUrlDiagnostic] = ingest_result.url_diagnostics
            selected: SavedSearchUrlDiagnostic | None = None
            candidate_domain = self._get_domain(url=candidate_url)
            for diag in diagnostics:
                if not diag.resolved_url:
                    continue
                if self._get_domain(url=diag.resolved_url) != candidate_domain:
                    continue
                selected = diag
                if (diag.parsed or 0) > 0:
                    break
            if selected is None and diagnostics:
                selected = diagnostics[0]

            return WebsiteSourceMetadata(
                source_name=ingest_result.source_name,
                resolved_url=selected.resolved_url if selected else None,
                render_http_status=selected.render_http_status if selected else None,
                render_error=selected.render_error if selected else None,
            )

        results = [
            SearchResult(
                url=candidate.url,
                domain=self._get_domain(url=candidate.url),
                search_engine_name=search_engine_name,
                website_source=_candidate_metadata(candidate.url),
                candidate_title=candidate.title,
                candidate_price=candidate.price,
                candidate_images=candidate.image_urls,
            )
            for candidate in ingest_result.candidates
        ]
        logger.debug(
            "Website-source=%s produced %s result URLs.",
            ingest_result.source_name,
            len(results),
        )
        return results


class Searcher(RedisCacher, DomainUtils):
    """Class to perform searches using different search engines."""

    _post_search_retry_stop_after = 3

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        serpapi_key: str,
        zyteapi_key: str,
        redis_use_cache: bool = REDIS_USE_CACHE,
    ):
        """Initializes the Search class with the given SerpAPI key.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            serpapi_key: The API key for SERP API.
            zyteapi_key: ZyteAPI key for fallback when direct access fails.
            redis_use_cache: Whether to use caching by a redis instance or not.
        """
        RedisCacher.__init__(self=self, use_cache=redis_use_cache)

        self._http_client = http_client
        self._google = SerpAPIGoogle(http_client=http_client, api_key=serpapi_key)
        self._google_shopping = SerpAPIGoogleShopping(
            http_client=http_client,
            api_key=serpapi_key,
        )
        self._toppreise = Toppreise(
            http_client=http_client,
            zyteapi_key=zyteapi_key,
            redis_use_cache=redis_use_cache,
        )
        self._saved_search_engine = WebsiteSearch(
            http_client=http_client,
            zyteapi_key=zyteapi_key,
            redis_use_cache=redis_use_cache,
        )
        self._search_handlers: Dict[
            SearchEngineName,
            Callable[..., Awaitable[List[SearchResult]]],
        ] = {
            SearchEngineName.GOOGLE: self._search_google,
            SearchEngineName.GOOGLE_SHOPPING: self._search_google_shopping,
            SearchEngineName.TOPPREISE: self._search_toppreise,
            SearchEngineName.WEBSITE_SOURCE: self._search_website_source,
        }
        self._post_search_enabled_engines = {
            SearchEngineName.GOOGLE,
            SearchEngineName.GOOGLE_SHOPPING,
            SearchEngineName.TOPPREISE,
        }

    async def _search_google(
        self,
        search_term: str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
        **_: object,
    ) -> List[SearchResult]:
        return await self._google.search(
            search_term=search_term,
            language=language,
            location=location,
            num_results=num_results,
            marketplaces=marketplaces,
        )

    async def _search_google_shopping(
        self,
        search_term: str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
        **_: object,
    ) -> List[SearchResult]:
        return await self._google_shopping.search(
            search_term=search_term,
            language=language,
            location=location,
            num_results=num_results,
            marketplaces=marketplaces,
        )

    async def _search_toppreise(
        self,
        search_term: str,
        language: Language,
        num_results: int,
        **_: object,
    ) -> List[SearchResult]:
        return await self._toppreise.search(
            search_term=search_term,
            language=language,
            num_results=num_results,
        )

    async def _search_website_source(
        self,
        search_term: str,
        num_results: int,
        website_source_source: WebsiteSource | None = None,
        **_: object,
    ) -> List[SearchResult]:
        if website_source_source is None:
            logger.warning(
                "search_engine='website_source' called without website_source_source; skipping."
            )
            return []
        return await self._saved_search_engine.search(
            source=website_source_source,
            search_term=search_term,
            num_results=num_results,
        )

    async def _post_search_google_shopping_immersive(self, url: str) -> List[str]:
        """Post-search for product URLs from a Google Shopping immersive product page.

        Args:
            url: The URL of the Google Shopping product page.
        """
        # Add SerpAPI key to the url
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api_key={self._google_shopping._api_key}"

        # Fetch the content of the Google Shopping product page
        response = await self._google_shopping.http_client_get(url=url)

        # Get external product urls from the data
        data = response.json()
        urls = self._google_shopping._extract_product_urls_from_immersive_product_api(
            data=data
        )
        return urls

    async def _post_search_toppreise_comparison(self, url: str) -> List[str]:
        """Post-search for product URLs from a Toppreise product comparison page.

        Note:
            In comparison to the function Toppreise._search, here we extract the urls from
            product comparison pages (f.e. https://www.toppreise.ch/preisvergleich/). These
            pages can also be found in the results of a google search.

        Args:
            url: The URL of the Toppreise product listing page.
        """
        # Perform the request with fallback if necessary
        content = await self._toppreise.http_client_get_with_fallback(url=url)

        # Get external product urls from the content
        urls = self._toppreise._extract_product_urls_from_comparison_page(
            content=content
        )
        return urls

    async def _post_search(self, results: List[SearchResult]) -> List[SearchResult]:
        """Post-search for additional embedded product URLs from the obtained results.

        Note:
            This function is used to extract embedded product URLs from
            product listing pages (e.g. Toppreise, Google Shopping) if needed.

        Args:
            results: The list of SearchResult objects obtained from the search.
        """
        post_search_results: List[SearchResult] = []
        for res in results:
            url = res.url
            post_search_urls: List[str] = []

            # Extract embedded product URLs from the Google Shopping immersive product page
            if "engine=google_immersive_product" in url:
                logger.debug(
                    f'Extracting embedded product URLs from url="{url}" found by search_engine="{res.search_engine_name}"'
                )
                post_search_urls = await self._post_search_google_shopping_immersive(
                    url=url
                )
                logger.debug(
                    f'Extracted {len(post_search_urls)} embedded product URLs from url="{url}".'
                )

            # Extract embedded product URLs from the Toppreise product listing page
            elif any(pth in url for pth in TOPPREISE_COMPARISON_PATHS):
                logger.debug(
                    f'Extracting embedded product URLs from url="{url}" found by search_engine="{res.search_engine_name}"'
                )
                post_search_urls = await self._post_search_toppreise_comparison(url=url)
                logger.debug(
                    f'Extracted {len(post_search_urls)} embedded product URLs from url="{url}".'
                )

            # Add the extracted product URLs as SearchResult objects
            psr = [
                SearchResult(
                    url=psu,
                    domain=self._get_domain(url=psu),
                    search_engine_name=res.search_engine_name,
                )
                for psu in post_search_urls
            ]
            post_search_results.extend(psr)

        return post_search_results

    @staticmethod
    def _domain_in_host(domain: str, host: Host) -> bool:
        """Checks if the domain is present in the host.

        Note:
            By checking `if domain == hst_dom or domain.endswith(f".{hst_dom}")`
            it also checks for subdomains. For example, if the domain is
            `link.springer.com` and the host domain is `springer.com`,
            it will be detected as being present in the hosts.

        Args:
            domain: The domain to check.
            host: The host to check against.
        """
        return any(
            domain == hst_dom or domain.endswith(f".{hst_dom}")
            for hst_dom in host.domains
        )

    def _domain_in_hosts(self, domain: str, hosts: List[Host]) -> bool:
        """Checks if the domain is present in the list of hosts.

        Args:
            domain: The domain to check.
            hosts: The list of hosts to check against.
        """
        return any(self._domain_in_host(domain=domain, host=hst) for hst in hosts)

    @staticmethod
    def _relevant_country_code(url: str, country_code: str) -> bool:
        """Determines whether the url shows relevant country codes.

        Args:
            url: The URL to investigate.
            country_code: The country code used to filter the products.
        """
        url = url.lower()
        country_code_relevance = f".{country_code}" in url
        default_relevance = any(cc in url for cc in SEARCH_DEFAULT_COUNTRY_CODES)
        return country_code_relevance or default_relevance

    def _is_excluded_url(self, domain: str, excluded_urls: List[Host]) -> bool:
        """Checks if the domain is in the excluded URLs.

        Args:
            domain: The domain to check.
            excluded_urls: The list of excluded URLs.
        """
        return self._domain_in_hosts(domain=domain, hosts=excluded_urls)

    def _apply_filters(
        self,
        result: SearchResult,
        location: Location,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> SearchResult:
        """Checks for filters and updates the SearchResult accordingly.

        Args:
            result: The SearchResult object to check.
            location: The location to use for the query.
            marketplaces: The list of marketplaces to compare the URL against.
            excluded_urls: The list of excluded URLs.
        """
        domain = result.domain
        # Check if the URL is in the marketplaces (if yes, keep the result un-touched)
        if marketplaces:
            if self._domain_in_hosts(domain=domain, hosts=marketplaces):
                return result

        # Check if the URL has a relevant country_code
        if not self._relevant_country_code(url=result.url, country_code=location.code):
            result.filtered = True
            result.filtered_at_stage = "Search (country code filtering)"
            return result

        # Check if the URL is in the excluded URLs
        if excluded_urls and self._is_excluded_url(result.domain, excluded_urls):
            result.filtered = True
            result.filtered_at_stage = "Search (excluded URLs filtering)"
            return result

        return result

    async def apply(
        self,
        search_term: str,
        search_engine: SearchEngineName | str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
        website_source_source: WebsiteSource | None = None,
        saved_search_source: WebsiteSource | None = None,
    ) -> List[SearchResult]:
        """Performs a search from given search engine."""
        if saved_search_source is not None:
            if website_source_source is not None:
                raise ValueError(
                    "Provide only one of website_source_source or saved_search_source."
                )
            logger.warning(
                "saved_search_source is deprecated; use website_source_source instead."
            )
            website_source_source = saved_search_source

        logger.info(
            f'Performing search for term="{search_term}" using engine="{search_engine}".'
        )

        # -------------------------------
        # SEARCH
        # -------------------------------
        # Map string to SearchEngineName if needed
        if isinstance(search_engine, str):
            search_engine = SearchEngineName(search_engine)

        search_handler = self._search_handlers.get(search_engine)
        if search_handler is None:
            raise ValueError(f"Unknown search engine: {search_engine}")
        results = await search_handler(
            search_term=search_term,
            language=language,
            location=location,
            num_results=num_results,
            marketplaces=marketplaces,
            website_source_source=website_source_source,
        )

        # -------------------------------
        # POST-SEARCH URL EXTRACTION
        # -------------------------------
        if search_engine in self._post_search_enabled_engines:
            post_search_results = await self._post_search(results=results)
            post_search_results = post_search_results[:num_results]
            results.extend(post_search_results)

        # -------------------------------
        # FILTERS
        # -------------------------------
        # Apply filters
        results = [
            self._apply_filters(
                result=res,
                location=location,
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
            )
            for res in results
        ]

        logger.info(
            f'Search for term="{search_term}" using engine="{search_engine}" produced {len(results)} results.'
        )
        return results
