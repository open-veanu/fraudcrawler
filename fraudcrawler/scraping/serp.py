import asyncio
from enum import Enum
import logging
from pydantic import BaseModel
from typing import List
from urllib.parse import urlparse

from fraudcrawler.settings import MAX_RETRIES, RETRY_DELAY, SERP_DEFAULT_COUNTRY_CODES
from fraudcrawler.base.base import Host, Language, Location, AsyncClient
import re

logger = logging.getLogger(__name__)


class SerpResult(BaseModel):
    """Model for a single search result from SerpApi."""

    url: str
    domain: str
    marketplace_name: str
    filtered: bool = False
    filtered_at_stage: str | None = None


class SearchEngine(Enum):
    """Enum for the supported search engines."""

    GOOGLE = "google"
    GOOGLE_SHOPPING = "google_shopping"


class SerpApi(AsyncClient):
    """A client to interact with the SerpApi for performing searches."""

    _endpoint = "https://serpapi.com/search"
    _engine_marketplace_names = {
        SearchEngine.GOOGLE.value: "Google",
        SearchEngine.GOOGLE_SHOPPING.value: "Google Shopping",
    }
    _hostname_pattern = r"^(?:https?:\/\/)?([^\/:?#]+)"

    def __init__(
        self,
        api_key: str,
        max_retries: int = MAX_RETRIES,
        retry_delay: int = RETRY_DELAY,
    ):
        """Initializes the SerpApiClient with the given API key.

        Args:
            api_key: The API key for SerpApi.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Delay between retries in seconds.
        """
        super().__init__()
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def _get_domain(self, url: str) -> str:
        """Extracts the second-level domain together with the top-level domain (e.g. `google.com`).

        Args:
            url: The URL to be processed.

        """
        # Add scheme (if needed -> urlparse requires it)
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
    def _extract_search_results(response: dict, engine: str) -> List[str]:
        """Extracts search results from the response based on the engine type.

        Args:
            response: The response from the SerpApi search.
            engine: The search engine used.

        Returns:
            A list of URLs extracted from the response.
        """
        urls = []
        if engine == SearchEngine.GOOGLE.value:
            # Get the organic_results
            results = response.get("organic_results")
            if results is None:
                logger.warning(f'No SerpAPI results for engine="{engine}".')
            else:
                urls = [url for res in results if (url := res.get("link"))]

        elif engine == SearchEngine.GOOGLE_SHOPPING.value:
            # Get the shopping_results
            results = response.get("shopping_results")
            if results is None:
                logger.warning(f'No SerpAPI results for engine="{engine}".')
            else:
                urls = [url for res in results if (url := res.get("product_link"))]

        else:
            raise ValueError(f"Invalid SerpAPI search engine: {engine}")

        return urls

    async def _search(
        self,
        engine: str,
        search_string: str,
        language: Language,
        location: Location,
        num_results: int,
    ) -> List[str]:
        """Performs a search using SerpApi and returns the URLs of the results.

        Args:
            engine: The search engine to use.
            search_string: The search string (with potentially added site: parameters).
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.

        The SerpAPI parameters are:
            engine: The search engine to use ('google' NOT 'google_shopping').
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
        if engine not in self._engine_marketplace_names:
            raise ValueError(
                f"Invalid SerpAPI search engine: {engine}. "
                f"Supported engines are: {list(self._engine_marketplace_names.keys())}."
            )
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

        # Perform the request
        attempts = 0
        err = None
        while attempts < self._max_retries:
            try:
                logger.debug(
                    f'Performing SerpAPI search with q="{search_string}" (Attempt {attempts + 1}).'
                )
                response = await self.get(url=self._endpoint, params=params)
                break
            except Exception as e:
                logger.error(f"SerpAPI search failed with error: {e}.")
                err = e
            attempts += 1
            if attempts < self._max_retries:
                await asyncio.sleep(self._retry_delay)
        if err is not None:
            raise err

        # Extract the URLs from the response
        urls = self._extract_search_results(response=response, engine=engine)

        logger.debug(
            f'Found total of {len(urls)} URLs from SerpApi search for q="{search_string}" and engine="{engine}".'
        )
        return urls

    @staticmethod
    def _relevant_country_code(url: str, country_code: str) -> bool:
        """Determines whether the url shows relevant country codes.

        Args:
            url: The URL to investigate.
            country_code: The country code used to filter the products.
        """
        url = url.lower()
        country_code_relevance = f".{country_code}" in url
        default_relevance = any(cc in url for cc in SERP_DEFAULT_COUNTRY_CODES)
        return country_code_relevance or default_relevance

    @staticmethod
    def _domain_in_host(domain: str, host: Host) -> bool:
        """Checks if the domain is present in the host.

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

        Note:
            By checking `if domain == hst_dom or domain.endswith(f".{hst_dom}")`
            it also checks for subdomains. For example, if the domain is
            `link.springer.com` and the host domain is `springer.com`,
            it will be detected as being present in the hosts.

        Args:
            domain: The domain to check.
            hosts: The list of hosts to check against.
        """
        return any(self._domain_in_host(domain=domain, host=hst) for hst in hosts)

    def _is_excluded_url(self, domain: str, excluded_urls: List[Host]) -> bool:
        """Checks if the domain is in the excluded URLs.

        Args:
            domain: The domain to check.
            excluded_urls: The list of excluded URLs.
        """
        return self._domain_in_hosts(domain=domain, hosts=excluded_urls)

    def _apply_filters(
        self,
        result: SerpResult,
        location: Location,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> SerpResult:
        """Checks for filters and updates the SerpResult accordingly.

        Args:
            result: The SerpResult object to check.
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
            result.filtered_at_stage = "SerpAPI (country code filtering)"
            return result

        # Check if the URL is in the excluded URLs
        if excluded_urls and self._is_excluded_url(result.domain, excluded_urls):
            result.filtered = True
            result.filtered_at_stage = "SerpAPI (excluded URLs filtering)"
            return result

        return result

    def _create_serp_result(
        self,
        engine: str,
        url: str,
        location: Location,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> SerpResult:
        """From a given url it creates the class:`SerpResult` instance.

        If marketplaces is None or the domain can not be extracted, the default marketplace name is used.

        Args:
            engine: The search engine used.
            url: The URL to be processed.
            location:  The location to use for the query.
            marketplaces: The list of marketplaces to compare the URL against.
            excluded_urls: The list of excluded URLs.
        """
        # Get marketplace name
        domain = self._get_domain(url=url)

        # Select marketplace name based on engine
        marketplace_name = self._engine_marketplace_names[engine]

        if marketplaces:
            try:
                marketplace_name = next(
                    mp.name
                    for mp in marketplaces
                    if self._domain_in_host(domain=domain, host=mp)
                )
            except StopIteration:
                logger.warning(f'Failed to find marketplace for domain="{domain}".')

        # Create the SerpResult object
        result = SerpResult(
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

    async def _search_google(
        self,
        search_string: str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> List[SerpResult]:
        """Performs a google search using SerpApi and returns SerpResults.

        Args:
            search_string: The search string (with potentially added site: parameters).
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
        """
        engine = SearchEngine.GOOGLE.value

        # Perform the search
        urls = await self._search(
            engine=engine,
            search_string=search_string,
            language=language,
            location=location,
            num_results=num_results,
        )

        # Create SerpResult objects from the URLs
        results = [
            self._create_serp_result(
                url=url,
                location=location,
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
                engine=engine,
            )
            for url in urls
        ]

        logger.debug(
            f'Produced {len(results)} results from google search with q="{search_string}".'
        )
        return results

    async def _search_google_shopping(
        self,
        search_string: str,
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> List[SerpResult]:
        """Performs a google search using SerpApi and returns SerpResults.

        Args:
            search_string: The search string (with potentially added site: parameters).
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
        """
        engine = SearchEngine.GOOGLE_SHOPPING.value

        # Perform the search
        urls = await self._search(
            engine=engine,
            search_string=search_string,
            language=language,
            location=location,
            num_results=num_results,
        )

        # !!! NOTE !!!: Google Shopping results do not properly support the 'num' parameter,
        # so we might get more results than requested. This is a known issue with SerpAPI
        # and Google Shopping searches (see https://github.com/serpapi/public-roadmap/issues/1858)
        urls = urls[:num_results]

        # Create SerpResult objects from the URLs
        results = [
            self._create_serp_result(
                url=url,
                location=location,
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
                engine=engine,
            )
            for url in urls
        ]

        logger.debug(
            f'Produced {len(results)} results from google shopping search with q="{search_string}".'
        )
        return results

    async def apply(
        self,
        search_term: str,
        search_engines: List[SearchEngine],
        language: Language,
        location: Location,
        num_results: int,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> List[SerpResult]:
        """Performs a search using SerpApi, filters based on country code and returns the URLs.

        Args:
            search_term: The search term to use for the query.
            language: The language to use for the query.
            location: The location to use for the query.
            num_results: Max number of results to return (default: 10).
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
        """
        # Setup the parameters
        logger.info(f'Performing SerpAPI search for search_term="{search_term}".')

        # Setup the search string
        search_string = search_term
        if marketplaces:
            sites = [dom for host in marketplaces for dom in host.domains]
            search_string += " site:" + " OR site:".join(s for s in sites)

        # Initialize the results list
        results: List[SerpResult] = []

        # Perform the google search
        if SearchEngine.GOOGLE in search_engines:
            ggl_res = await self._search_google(
                search_string=search_string,
                language=language,
                location=location,
                num_results=num_results,
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
            )
            results.extend(ggl_res)

        # Perform the google shopping search
        if SearchEngine.GOOGLE_SHOPPING in search_engines:
            shp_res = await self._search_google_shopping(
                search_string=search_string,
                language=language,
                location=location,
                num_results=num_results,
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
            )
            results.extend(shp_res)

        num_non_filtered = len([res for res in results if not res.filtered])
        logger.info(
            f'Produced a total of {num_non_filtered} results from SerpApi search with q="{search_string}".'
        )
        return results
