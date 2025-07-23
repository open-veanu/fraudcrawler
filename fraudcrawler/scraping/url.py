import logging
from typing import List, Set, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, quote, urlunparse, ParseResult

from fraudcrawler.settings import KNOWN_TRACKERS

logger = logging.getLogger(__name__)


class URLCollector:
    """A class to collect and de-duplicate URLs."""

    def __init__(self):
        self.collected_currently: Set[str] = set()
        self.collected_previously: Set[str] = set()

    @staticmethod
    def remove_tracking_parameters(url: str) -> str:
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
        if remove_all:
            filtered_queries = []
        else:
            filtered_queries = [
                q
                for q in queries
                if not any(q[0].startswith(tracker) for tracker in KNOWN_TRACKERS)
            ]

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
