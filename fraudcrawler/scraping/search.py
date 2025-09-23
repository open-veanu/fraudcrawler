from abc import ABC, abstractmethod
from enum import Enum
import logging
from pydantic import BaseModel
from typing import Dict, List
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
from bs4.element import Tag
import httpx
from tenacity import RetryCallState

from fraudcrawler.settings import (
    SEARCH_DEFAULT_COUNTRY_CODES,
    TOPPREISE_SEARCH_PATHS,
    TOPPREISE_COMPARISON_PATHS,
)
from fraudcrawler.base.base import Host, Language, Location, DomainUtils
from fraudcrawler.base.retry import get_async_retry
from fraudcrawler.scraping.zyte import ZyteAPI

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Model for a single search result."""

    url: str
    domain: str
    search_engine_name: str
    filtered: bool = False
    filtered_at_stage: str | None = None


class SearchEngineName(Enum):
    """Enum for search engine names."""

    GOOGLE = "google"
    GOOGLE_SHOPPING = "google_shopping"
    TOPPREISE = "toppreise"


class SearchEngine(ABC, DomainUtils):
    """Abstract base class for search engines."""

    _hostname_pattern = r"^(?:https?:\/\/)?([^\/:?#]+)"

    def __init__(self, http_client: httpx.AsyncClient):
        """Initializes the SearchEngine with the given HTTP client.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
        """
        self._http_client = http_client

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
            num: The number of results to return.
            api_key: The API key to use for the search.
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
            "num": num_results,
            "api_key": self._api_key,
        }
        logger.debug(f"SerpAPI search with params: {params}")

        # Perform the search request
        response: httpx.Response = await self.http_client_get(
            url=self._endpoint, params=params
        )

        # Extract the URLs from the response
        data = response.json()
        urls = self._extract_search_results_urls(data=data)

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

    def __init__(self, http_client: httpx.AsyncClient, zyteapi_key: str):
        """Initializes the Toppreise client.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            zyteapi_key: ZyteAPI key for fallback when direct access fails.
        """
        super().__init__(http_client=http_client)
        self._zyteapi = ZyteAPI(http_client=http_client, api_key=zyteapi_key)

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
                    logger.error(msg)
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


class Searcher(DomainUtils):
    """Class to perform searches using different search engines."""

    _post_search_retry_stop_after = 3

    def __init__(
        self, http_client: httpx.AsyncClient, serpapi_key: str, zyteapi_key: str
    ):
        """Initializes the Search class with the given SerpAPI key.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            serpapi_key: The API key for SERP API.
            zyteapi_key: ZyteAPI key for fallback when direct access fails.
        """
        self._http_client = http_client
        self._google = SerpAPIGoogle(http_client=http_client, api_key=serpapi_key)
        self._google_shopping = SerpAPIGoogleShopping(
            http_client=http_client,
            api_key=serpapi_key,
        )
        self._toppreise = Toppreise(
            http_client=http_client,
            zyteapi_key=zyteapi_key,
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
    ) -> List[SearchResult]:
        """Performs a search and returns SearchResults.

        Args:
            search_term: The search term to use for the query.
            search_engine: The search engine to use for the search.
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results per search engine.
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
        """
        logger.info(
            f'Performing search for term="{search_term}" using engine="{search_engine}".'
        )

        # -------------------------------
        # SEARCH
        # -------------------------------
        # Map string to SearchEngineName if needed
        if isinstance(search_engine, str):
            search_engine = SearchEngineName(search_engine)

        # Make SerpAPI google search
        if search_engine == SearchEngineName.GOOGLE:
            results = await self._google.search(
                search_term=search_term,
                language=language,
                location=location,
                num_results=num_results,
                marketplaces=marketplaces,
            )

        # Make SerpAPI google shopping search
        elif search_engine == SearchEngineName.GOOGLE_SHOPPING:
            results = await self._google_shopping.search(
                search_term=search_term,
                language=language,
                location=location,
                num_results=num_results,
                marketplaces=marketplaces,
            )

        # Make Toppreise search
        elif search_engine == SearchEngineName.TOPPREISE:
            results = await self._toppreise.search(
                search_term=search_term,
                language=language,
                num_results=num_results,
            )

        # Other search engines can be added here (raise unknown engine error otherwise)
        else:
            raise ValueError(f"Unknown search engine: {search_engine}")

        # -------------------------------
        # POST-SEARCH URL EXTRACTION
        # -------------------------------
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
