import asyncio
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


class SerpApi(AsyncClient):
    """A client to interact with the SerpApi for performing searches."""

    _endpoint = "https://serpapi.com/search"
    _engine = "google"
    _default_marketplace_name = "Google"
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

    async def _search(
        self,
        search_string: str,
        language: Language,
        location: Location,
        num_results: int,
    ) -> List[str]:
        """Performs a search using SerpApi and returns the URLs of the results.

        Args:
            search_string: The search string (with potentially added site: parameters).
            language: The language to use for the query ('hl' parameter).
            location: The location to use for the query ('gl' parameter).
            num_results: Max number of results to return.

        The SerpAPI parameters are:
            engine: The search engine to use ('google' NOT 'google_shopping').
            q: The search string (with potentially added site: parameters).
            google_domain: The Google domain to use for the search (e.g. google.[com]).
            location_[requested|used]: The location to use for the search.
            tbs: The time-based search parameters (e.g. 'ctr:CH&cr:countryCH').
            gl: The country code to use for the search.
            hl: The language code to use for the search.
            num: The number of results to return.
            api_key: The API key to use for the search.
        """
        # Setup the parameters
        params = {
            "engine": self._engine,
            "q": search_string,
            "google_domain": f"google.{location.code}",
            "location_requested": location.name,
            "location_used": location.name,
            "tbs": f"ctr:{location.code.upper()}&cr:country{location.code.upper()}",
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

        # Get the organic_results
        results = response.get("organic_results")
        if results is None:
            logger.warning(
                f'No organic_results key in SerpAPI results for search_string="{search_string}".'
            )
            return []

        # Extract urls
        urls = [res.get("link") for res in results]
        logger.debug(
            f'Found {len(urls)} URLs from SerpApi search for q="{search_string}".'
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
        url: str,
        location: Location,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> SerpResult:
        """From a given url it creates the class:`SerpResult` instance.

        If marketplaces is None or the domain can not be extracted, the default marketplace name is used.

        Args:
            url: The URL to be processed.
            location:  The location to use for the query.
            marketplaces: The list of marketplaces to compare the URL against.
        """
        # Get marketplace name
        domain = self._get_domain(url=url)
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

    async def apply(
        self,
        search_term: str,
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

        # Perform the search
        urls = await self._search(
            search_string=search_string,
            language=language,
            location=location,
            num_results=num_results,
        )

        # Form the SerpResult objects
        results = [
            self._create_serp_result(
                url=url,
                location=location,
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
            )
            for url in urls
        ]

        num_non_filtered = len([res for res in results if not res.filtered])
        logger.info(
            f'Produced {num_non_filtered} results from SerpApi search with q="{search_string}".'
        )
        return results
