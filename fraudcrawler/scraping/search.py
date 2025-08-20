from abc import ABC, abstractmethod
from enum import Enum
import logging
from pydantic import BaseModel
import re
import requests
from typing import List
from urllib.parse import urlparse, quote_plus

from bs4 import BeautifulSoup
from tenacity import RetryCallState

from fraudcrawler.settings import SEARCH_DEFAULT_COUNTRY_CODES
from fraudcrawler.base.base import Host, Language, Location, AsyncClient
from fraudcrawler.base.retry import get_async_retry, get_sync_retry

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Model for a single search result."""

    url: str
    domain: str
    marketplace_name: str
    filtered: bool = False
    filtered_at_stage: str | None = None


class SearchEngineName(Enum):
    """Enum for search engine names."""
    GOOGLE = "google"
    GOOGLE_SHOPPING = "google_shopping"
    TOPPREISE = "toppreise"


class SearchEngine(ABC, AsyncClient):
    """Abstract base class for search engines."""
    _hostname_pattern = r"^(?:https?:\/\/)?([^\/:?#]+)"

    def __init__(self, default_marketplace_name: str):
        """Initializes the search engine."""
        self._default_marketplace_name = default_marketplace_name

    @abstractmethod
    async def apply(self, **kwargs) -> List[SearchResult]:
        """Apply the search with the given parameters and return results."""
        pass

    @classmethod
    def _log_before(cls, search_string: str, retry_state: RetryCallState | None) -> None:
        """Context aware logging before the request is made."""
        if retry_state:
            logger.debug(
                f'Performing search in {cls.__name__} with q="{search_string}" '
                f"(attempt {retry_state.attempt_number})."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before.")

    @classmethod
    def _log_before_sleep(
        cls, search_string: str, retry_state: RetryCallState | None
    ) -> None:
        """Context aware logging before sleeping after a failed request."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f'Attempt {retry_state.attempt_number} of {cls.__name__} search with q="{search_string}" '
                f"failed with error: {retry_state.outcome.exception()}. "
                f"Retrying in {retry_state.upcoming_sleep:.0f} seconds."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before_sleep.")

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

    def _create_search_result(
        self,
        url: str,
        location: Location,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> SearchResult:
        """From a given url it creates the class:`SearchResult` instance.

        If marketplaces is None or the domain can not be extracted, the default marketplace name is used.

        Args:
            url: The URL to be processed.
            location:  The location to use for the query.
            marketplaces: The list of marketplaces to compare the URL against.
            excluded_urls: The list of excluded URLs.
        """
        # Get marketplace name
        domain = self._get_domain(url=url)

        # Set marketplace name (default if not found)
        marketplace_name = self._default_marketplace_name
        if marketplaces:
            try:
                marketplace_name = next(
                    mp.name
                    for mp in marketplaces
                    if self._domain_in_host(domain=domain, host=mp)
                )
            except StopIteration:
                logger.warning(f'Failed to find marketplace for domain="{domain}".')

        # Create the SearchResult object
        result = SearchResult(
            url=url,
            domain=domain,
            marketplace_name=marketplace_name,
        )

        # Apply filters
        result = self._apply_filters(
            result=result,
            location=location,
            marketplaces=marketplaces,
            excluded_urls=excluded_urls,
        )
        return result


class SerpAPI(SearchEngine):
    """Base class for SerpAPI search engines."""

    _endpoint = "https://serpapi.com/search"

    def __init__(self, api_key: str):
        """Initializes the SerpAPI client with the given API key.

        Args:
            api_key: The API key for SerpAPI.
        """
        default_marketplace_name = self._engine.replace("_", " ").title()
        super().__init__(default_marketplace_name=default_marketplace_name)
        self._api_key = api_key

    @property
    @abstractmethod
    def _engine(self) -> str:
        """The search engine name used in the SerpAPI request."""
        pass

    @staticmethod
    @abstractmethod
    def _extract_search_results_urls(response: dict) -> List[str]:
        """Extracts search results urls from the response.

        Args:
            response: The response from the SerpAPI search.
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

        # Setup the parameters
        params = {
            "engine": engine,
            "q": search_string,
            "google_domain": f"google.{location.code}",
            "location_requested": location.name,
            "location_used": location.name,
            "tbs": f"ctr:{location.code.upper()}",
            "cr": f"country{location.code.upper()}",
            "gl": location.code,
            "hl": language.code,
            "num": num_results,
            "api_key": self._api_key,
        }
        logger.debug(f"SerpAPI search with params: {params}")

        # Perform the request and retry if necessary. There is some context aware logging:
        #  - `before`: before the request is made (and before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            search_string=search_string, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            search_string=search_string, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                response = await self.get(url=self._endpoint, params=params)

        # Extract the URLs from the response
        urls = self._extract_search_results_urls(response=response)

        logger.debug(
            f'Found total of {len(urls)} URLs from SerpAPI search for q="{search_string}" and engine="{engine}".'
        )
        return urls


class SerpAPIGoogle(SerpAPI):
    """Search engine for Google in SerpAPI."""

    def __init__(self, api_key: str):
        """Initializes the SerpAPIGoogle client with the given API key.

        Args:
            api_key: The API key for SerpAPI.
        """
        super().__init__(api_key=api_key)

    @property
    def _engine(self) -> str:
        """The search engine name used in the SerpAPI request."""
        return 'google'
    
    @staticmethod
    def _extract_search_results_urls(response: dict) -> List[str]:
        """Extracts search results urls from the response.

        Args:
            response: The response from the SerpApi search.
        """
        results = response.get("organic_results")
        if results is not None:
            return [url for res in results if (url := res.get("link"))]
        return []
    
    async def apply(
        self,
        search_term: str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> List[SearchResult]:
        """Performs a google search using SerpApi and returns SearchResults.

        Args:
            search_term: The search term to use for the query.
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
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

        # Create SearchResult objects from the URLs
        results = [
            self._create_search_result(
                url=url,
                location=location,
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
            )
            for url in urls
        ]

        logger.debug(
            f'Produced {len(results)} results from SerpAPI with engine="{self._engine}" and q="{search_string}".'
        )
        return results


class SerpAPIGoogleShopping(SerpAPI):
    """Search engine for Google Shopping in SerpAPI."""

    def __init__(self, api_key: str):
        """Initializes the SerpAPIGoogleShopping client with the given API key.

        Args:
            api_key: The API key for SerpAPI.
        """
        super().__init__(api_key=api_key)

    @property
    def _engine(self) -> str:
        """The search engine name used in the SerpAPI request."""
        return 'google_shopping'

    @staticmethod
    def _extract_search_results_urls(response: dict) -> List[str]:
        """Extracts search results urls from the response.

        Args:
            response: The response from the SerpApi search.
        """
        results = response.get("shopping_results")
        if results is not None:
            return [url for res in results if (url := res.get("product_link"))]
        return []

    async def apply(
        self,
        search_term: str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> List[SearchResult]:
        """Performs a google shopping search using SerpApi and returns SearchResults.

        Args:
            search_term: The search term to use for the query.
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
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

        # !!! NOTE !!!: Google Shopping results do not properly support the 'num' parameter,
        # so we might get more results than requested. This is a known issue with SerpAPI
        # and Google Shopping searches (see https://github.com/serpapi/public-roadmap/issues/1858)
        urls = urls[:num_results]

        # Create SearchResult objects from the URLs
        results = [
            self._create_search_result(
                url=url,
                location=location,
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
            )
            for url in urls
        ]

        logger.debug(
            f'Produced {len(results)} results from SerpAPI with engine="{self._engine}" and q="{search_string}".'
        )
        return results


class Toppreise(SearchEngine):
    """Search engine for toppreise.ch."""

    _default_marketplace_name = "Toppreise"
    _endpoint = "https://www.toppreise.ch/produktsuche"
    _timeout = 6  # seconds
    _max_redirects = 3
    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    def __init__(self):
        """Initializes the Toppreise search engine."""
        super().__init__(default_marketplace_name=self._default_marketplace_name)

    @staticmethod
    def _external_product_urls(links: List[str]) -> List[str]:
        """Filters the links to only include those that are external product links and normalizes urls."""
        hrefs = [
            href for link in links if (
                hasattr(link, "get")                    # Ensure we have a Tag object with href attribute
                and (href := link.get("href"))          # Ensure href is not None
                and not href.startswith("javascript:")  # Skip javascript links
                and isinstance(href, str)               # Ensure href is a string
                and "ext_" in href                      # Skip links that are not external product link
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
        return list(set(urls))
        
    def _get_external_product_urls(self, content: bytes) -> List[str]:
        """Extracts external product URLs from the Toppreise search results page.

        Args:
            content: The HTML content of the search results page.
        """
        # Parse the HTML
        soup = BeautifulSoup(content, "html.parser")
        links = soup.find_all("a", href=True)
        urls = self._external_product_urls(links=links)
        logger.debug(
            f"Found {len(urls)} external product URLs from Toppreise search results."
        )
        return urls

    def _resolve_redirects_safely(self, urls: List[str]) -> List[str]:
        """Resolves redirects for the given URLs and returns the final URLs."""
        # Find all the resolved URLs
        product_urls = []
        with requests.Session() as session:
            session.max_redirects = self._max_redirects
            for url in urls:
                try:
                    response = session.head(
                        url,
                        allow_redirects=True,
                        timeout=self._timeout,
                        headers=self._headers,
                    )
                    product_urls.append(response.url)

                except Exception as e:
                    logger.debug(f"Failed to resolve redirect for {url}: {e}")
                    product_urls.append(url)
        
        # Remove duplicates and return
        product_urls = list(set(product_urls))
        logger.debug(
            f"Resolved {len(product_urls)} product URLs from Toppreise search results."
        )
        return product_urls

    async def _search(self, search_string: str, num_results: int) -> List[str]:
        """Performs a search on Toppreise and returns the URLs of the results.
        
        Args:
            search_string: The search string to use for the query.
            num_results: Max number of results to return.
        """
        # Build the search URL for Toppreise
        encoded_search = quote_plus(search_string)
        url = f"{self._endpoint}?q={encoded_search}"
        logger.debug(f"Toppreise search URL: {url}")

        # Perform the request and retry if necessary. There is some context aware logging:
        #  - `before`: before the request is made (and before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            search_string=search_string, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            search_string=search_string, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                content = await self.get(
                    url=url,
                    headers=self._headers,
                    answer_format="bytes",
                )
        
        # Get external product urls from the content
        urls = self._get_external_product_urls(content=content)
        urls = urls[:num_results]  # Limit to num_results if needed

        # Resolve redirects for the URLs
        urls = self._resolve_redirects_safely(urls=urls)
        return urls

    async def apply(
        self,
        search_term: str,
        num_results: int,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> List[SearchResult]:
        """Performs a Toppreise search and returns SearchResults.

        Args:
            search_term: The search term to use for the query.
            num_results: Max number of results to return.
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
        """
        # Perform the search
        urls = await self._search(
            search_string=search_term,
            num_results=num_results,
        )

        # Create SearchResult objects from the URLs
        results = [
            self._create_search_result(
                url=url,
                location=Location(name="Switzerland", code="CH"),  # Toppreise is for Switzerland
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
            )
            for url in urls
        ]

        logger.debug(
            f'Produced {len(results)} results from Toppreise search with q="{search_term}".'
        )
        return results
        

class Search:
    """Class to perform searches using different search engines."""

    def __init__(self, serpapi_key):
        """Initializes the Search class with the given SerpAPI key.

        Args:
            serpapi_key: The API key for SERP API.
        """
        self._google = SerpAPIGoogle(api_key=serpapi_key)
        self._google_shopping = SerpAPIGoogleShopping(api_key=serpapi_key)
        self._toppreise = Toppreise()

    async def apply(
        self,
        search_string: str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
        search_engine_names: List[SearchEngineName | str] | None = None,
    ) -> List[SearchResult]:
        """Performs a search and returns SearchResults.

        Args:
            search_string: The search string (with potentially added site: parameters).
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
        """
        if search_engine_names is None:
            search_engine_names = list(SearchEngineName)
        else:
            search_engine_names = [
                SearchEngineName(sen) if isinstance(sen, str) else sen
                for sen in search_engine_names
            ]
        pass


#   ------------------------------------------------------------------
#   TODO: Remove the following commented out code once the SerpApi class is fully implemented.
#   ------------------------------------------------------------------

# class SerpApi(AsyncClient):
#     """A client to interact with the SerpApi for performing searches."""

#     _engine_marketplace_names = {
#         SearchEngine.GOOGLE.value: "Google",
#         SearchEngine.GOOGLE_SHOPPING.value: "Google Shopping",
#         SearchEngine.TOPPREISE.value: "Toppreise",
#     }

#     async def _search_toppreise(
#         self,
#         search_string: str,
#         location: Location,
#         num_results: int,
#         marketplaces: List[Host] | None = None,
#         excluded_urls: List[Host] | None = None,
#     ) -> List[SearchResult]:
#         """Performs a Toppreise search and returns SerpResults.

#         Args:
#             search_string: The search string (with potentially added site: parameters).
#             location: The location to use for the query (not used for Toppreise).
#             num_results: Max number of results to return.
#             marketplaces: The marketplaces to include in the search.
#             excluded_urls: The URLs to exclude from the search.
#         """
#         engine = SearchEngine.TOPPREISE.value

#         # Build the search URL for Toppreise
#         base_url = "https://www.toppreise.ch/produktsuche"
#         encoded_search = quote(search_string)
#         params = {"q": encoded_search, "cid": ""}
#         query_string = "&".join([f"{k}={v}" for k, v in params.items() if v])
#         search_url = f"{base_url}?{query_string}"

#         # Set headers to mimic a real browser
#         headers = {
#             "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
#             "Accept-Language": "en-US,en;q=0.5",
#             "Accept-Encoding": "gzip, deflate",
#             "Connection": "keep-alive",
#             "Upgrade-Insecure-Requests": "1",
#         }

#         try:
#             # Use aiohttp directly to handle HTML response
#             import aiohttp

#             async with aiohttp.ClientSession(headers=headers) as session:
#                 async with session.get(search_url) as response:
#                     response.raise_for_status()
#                     content = await response.read()

#             # Parse the HTML
#             soup = BeautifulSoup(content, "html.parser")

#             # Find all <a> tags and extract URLs
#             urls = []
#             for link in soup.find_all("a", href=True):
#                 # Ensure we have a Tag object with href attribute
#                 if not hasattr(link, "get"):
#                     continue

#                 href = link.get("href")

#                 # Skip empty links and javascript links
#                 if href and not href.startswith("javascript:"):
#                     # Ensure href is a string
#                     if not isinstance(href, str):
#                         continue

#                     # Make relative URLs absolute
#                     if href.startswith("/"):
#                         href = f"https://www.toppreise.ch{href}"
#                     elif not href.startswith("http"):
#                         href = f"https://www.toppreise.ch/{href}"

#                     # Look for external product links (preisvergleich pages)
#                     if "ext_" in href:
#                         # Try to resolve the redirect URL using the retry logic
#                         resolved_url = self._resolve_redirect_safely(href)
#                         if resolved_url:
#                             urls.append(resolved_url)
#                         else:
#                             # Fallback to original URL if resolution fails
#                             urls.append(href)

#             # Limit to requested number of results
#             urls = urls[:num_results]

#         except Exception as e:
#             logger.error(f"Error during Toppreise search: {e}")
#             urls = []

#         # Create SerpResult objects from the URLs
#         results = [
#             self._create_serp_result(
#                 url=url,
#                 location=location,
#                 marketplaces=marketplaces,
#                 excluded_urls=excluded_urls,
#                 engine=engine,
#             )
#             for url in urls
#         ]

#         logger.debug(
#             f'Produced {len(results)} results from Toppreise search with q="{search_string}".'
#         )
#         return results

#     def _resolve_redirect_safely(self, url: str, max_redirects: int = 3) -> str | None:
#         """
#         Safely resolve redirect URLs using the existing retry logic to avoid getting blocked.
#         Returns the resolved URL or None if resolution fails.
#         """
#         try:
#             # Use the same retry logic as the rest of the application
#             retry = get_sync_retry()

#             def _make_request():
#                 # Create a session with redirect limits
#                 session = requests.Session()
#                 session.max_redirects = max_redirects

#                 response = session.head(
#                     url,
#                     allow_redirects=True,
#                     timeout=6,
#                     headers={
#                         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
#                     },
#                 )
#                 return response.url

#             # Execute with retry logic - use __call__ method
#             resolved_url = retry(_make_request)
#             return resolved_url

#         except Exception as e:
#             logger.debug(f"Failed to resolve redirect for {url}: {e}")
#             return None

#     async def apply(
#         self,
#         search_term: str,
#         search_engines: List[SearchEngine],
#         language: Language,
#         location: Location,
#         num_results: int,
#         marketplaces: List[Host] | None = None,
#         excluded_urls: List[Host] | None = None,
#     ) -> List[SearchResult]:
#         """Performs a search using SerpApi, filters based on country code and returns the URLs.

#         Args:
#             search_term: The search term to use for the query.
#             language: The language to use for the query.
#             location: The location to use for the query.
#             num_results: Max number of results to return (default: 10).
#             marketplaces: The marketplaces to include in the search.
#             excluded_urls: The URLs to exclude from the search.
#         """
#         # Setup the parameters
#         logger.info(f'Performing SerpAPI search for search_term="{search_term}".')

#         # Setup the search string
#         search_string = search_term
#         if marketplaces:
#             sites = [dom for host in marketplaces for dom in host.domains]
#             search_string += " site:" + " OR site:".join(s for s in sites)

#         # Initialize the results list
#         results: List[SearchResult] = []

#         # Perform the google search
#         if SearchEngine.GOOGLE in search_engines:
#             ggl_res = await self._search_google(
#                 search_string=search_string,
#                 language=language,
#                 location=location,
#                 num_results=num_results,
#                 marketplaces=marketplaces,
#                 excluded_urls=excluded_urls,
#             )
#             results.extend(ggl_res)

#         # Perform the google shopping search
#         if SearchEngine.GOOGLE_SHOPPING in search_engines:
#             shp_res = await self._search_google_shopping(
#                 search_string=search_string,
#                 language=language,
#                 location=location,
#                 num_results=num_results,
#                 marketplaces=marketplaces,
#                 excluded_urls=excluded_urls,
#             )
#             results.extend(shp_res)

#         # Perform the Toppreise search
#         if SearchEngine.TOPPREISE in search_engines:
#             top_res = await self._search_toppreise(
#                 search_string=search_string,
#                 location=location,
#                 num_results=num_results,
#                 marketplaces=marketplaces,
#                 excluded_urls=excluded_urls,
#             )
#             results.extend(top_res)

#         num_non_filtered = len([res for res in results if not res.filtered])
#         logger.info(
#             f'Produced a total of {num_non_filtered} results from SerpApi search with q="{search_string}".'
#         )
#         return results

