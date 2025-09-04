from copy import deepcopy
import logging
from typing import List, Set, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, quote, urlunparse, ParseResult

from fraudcrawler.settings import KNOWN_TRACKERS
from fraudcrawler.base.base import ProductItem

logger = logging.getLogger(__name__)


class URLCollector:
    """A class to collect and de-duplicate URLs.
    
    Note:
        It might happen that a search engine returns URLs for pages with multiple product listings
        (e.g. Toppreise, Google Shopping). In this case, the URLCollector.apply method will extract
        the individual product URLs. This step is search engine specific and needs to be implemented
        individually.

        The reason why it is not implemented in the search engine classes is that such links might arise
        from any other search engine as well (i.e. Google Search can produce Toppreise links as well).
    """

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
    
    def _apply_deduplication(self, product: ProductItem) -> ProductItem:
        """Apply deduplication to a ProductItem based on its URL."""
        # Remove tracking parameters from the URL
        url = self._remove_tracking_parameters(product.url)
        product.url = url

        # deduplicate on current run
        if url in self._collected_currently:
            product.filtered = True
            product.filtered_at_stage = (
                "URL collection (current run deduplication)"
            )
            logger.debug(f"URL {url} already collected in current run")

        # deduplicate on previous runs coming from a db
        elif url in self._collected_previously:
            product.filtered = True
            product.filtered_at_stage = (
                "URL collection (previous run deduplication)"
            )
            logger.debug(f"URL {url} as already collected in previous run")
        
        # Add to currently collected URLs
        else:
            self._collected_currently.add(url)

        return product
    

    # def _get_toppreise_embedded_product_urls_(self, url: str) -> List[str]:
    #     """Extract embedded product URLs from a Toppreise product listing page.
        
    #     Note:
    #         In comparison to the function Toppreise._extract_search_results_urls, here
    #         we extract the urls from the product comparison table (https://www.toppreise.ch/preisvergleich/).
    #     """
    #     pass


    # def _get_embedded_product_urls(self, url: str) -> List[str]:
    #     """Extract embedded product URLs for predefined product listing pages."""
    #     pass

    async def apply(self, product: ProductItem) -> List[ProductItem] | None:
        """Collect all the relevant ProductItems from a given URL.
        
        Note:
            This function handles a given ProductItem with respect to:
                - Removing tracking parameters from the URL
                - Extracting individual product URLs if the URL points to a page with multiple products items
                  (as e.g. Toppreise, Google Shopping)
                - Check if the product has been collected yet based on the cleaned URL

        Args:
            product: The ProductItem to process.
        """
        logger.debug(f"Applying URL processing to product: {product.url}")

        # Check if the product has already been collected
        product = self._apply_deduplication(product=product)
        if product.filtered:
            logger.debug(f'Product {product.url} filtered after deduplication')
            return [product]

        # Extract embedded product URLs if applicable
        urls = self._get_embedded_product_urls(url=product.url)
        if not urls:
            logger.debug(f"No embedded product URLs extracted from {product.url}")
            return [product]

        # Create new ProductItems for each extracted URL
        logger.debug(f"Extracted {len(urls)} embedded product URLs from {product.url}")
        products: List[ProductItem] = []
        for url in urls:
            # Create new ProductItem
            prod = deepcopy(product)
            url = self._remove_tracking_parameters(url)
            prod.url = url

            # Check if the product has already been collected
            prod = self._apply_deduplication(product=prod)
            if not prod.filtered:
                self._collected_currently.add(url)
            products.append(prod)
        
        return products
