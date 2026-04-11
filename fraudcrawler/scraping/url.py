import logging
from typing import List, Set, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, quote, urlunparse, ParseResult

from fraudcrawler.settings import KNOWN_TRACKERS
from fraudcrawler.base.base import ProductItem

logger = logging.getLogger(__name__)


def should_drop_tracking_query_parameter(param_key: str) -> bool:
    """Returns True if a query parameter key is considered a tracker."""
    key = str(param_key or "").lower()
    return any(key.startswith(tracker) for tracker in KNOWN_TRACKERS)


def filter_tracking_query_entries(
    queries: List[Tuple[str, str]],
    *,
    remove_all: bool = False,
) -> List[Tuple[str, str]]:
    """Filter tracking query entries from an already parsed query list."""
    if remove_all:
        return []
    return [
        query for query in queries if not should_drop_tracking_query_parameter(query[0])
    ]


class URLCollector:
    """A class to collect and de-duplicate URLs."""

    def __init__(self):
        self._collected_currently: Set[str] = set()
        self._collected_previously: Set[str] = set()

    def add_previously_collected_urls(self, urls: List[str]) -> None:
        """Add a set of previously collected URLs to the internal state.

        Args:
            urls: A set of URLs that have been collected in previous runs.
        """
        self._collected_previously.update(urls)

    @staticmethod
    def _remove_tracking_parameters(url: str) -> str:
        """Remove tracking parameters from URLs.

        Args:
            url: The URL to clean.

        Returns:
            The cleaned URL without tracking parameters.
        """
        logging.debug(f"Removing tracking parameters from URL: {url}")

        # Parse the url
        parsed_url = urlparse(url)

        # Parse query parameters
        queries: List[Tuple[str, str]] = parse_qsl(
            parsed_url.query, keep_blank_values=True
        )
        remove_all = url.startswith(
            "https://www.ebay"
        )  # eBay URLs have all query parameters as tracking parameters
        filtered_queries = filter_tracking_query_entries(
            queries=queries, remove_all=remove_all
        )

        # Rebuild the URL without tracking parameters
        clean_url = ParseResult(
            scheme=parsed_url.scheme,
            netloc=parsed_url.netloc,
            path=parsed_url.path,
            params=parsed_url.params,
            query=urlencode(filtered_queries, quote_via=quote),
            fragment=parsed_url.fragment,
        )
        return urlunparse(clean_url)

    async def apply(self, product: ProductItem) -> ProductItem:
        """Manages the collection and deduplication of ProductItems.

        Args:
            product: The product item to process.
        """
        logger.debug(f'Processing product with  url="{product.url}"')

        # Remove tracking parameters from the URL
        url = self._remove_tracking_parameters(product.url)
        product.url = url

        # deduplicate on current run
        if url in self._collected_currently:
            product.filtered = True
            product.filtered_at_stage = "URL collection (current run deduplication)"
            logger.debug(f"URL {url} already collected in current run")

        # deduplicate on previous runs coming from a db
        elif url in self._collected_previously:
            product.filtered = True
            product.filtered_at_stage = "URL collection (previous run deduplication)"
            logger.debug(f"URL {url} as already collected in previous run")

        # Add to currently collected URLs
        else:
            self._collected_currently.add(url)

        return product
