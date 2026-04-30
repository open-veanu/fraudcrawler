from abc import ABC, abstractmethod
import hashlib
import logging
from typing import List, Set, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, quote, urlunparse, ParseResult

from aiocache.backends.redis import RedisBackend

from fraudcrawler.base.base import FilteredAtStage, ProductItem
from fraudcrawler.cache.cacher import RedisConfig
from fraudcrawler.settings import KNOWN_TRACKERS

logger = logging.getLogger(__name__)


def _should_drop_tracking_query_parameter(param_key: str) -> bool:
    """Return True if a query parameter key matches a known tracking prefix.

    Args:
        param_key: Query parameter key to test (case-insensitive).
    """
    key = str(param_key or "").lower()
    return any(key.startswith(tracker) for tracker in KNOWN_TRACKERS)


def filter_tracking_query_entries(
    queries: List[Tuple[str, str]],
    remove_all: bool = False,
) -> List[Tuple[str, str]]:
    """Filter tracking query entries from an already parsed query list.

    Args:
        queries: Parsed query parameters as (key, value) pairs.
        remove_all: If True, return an empty list regardless of content.
    """
    if remove_all:
        return []
    return [
        (key, val)
        for key, val in queries
        if not _should_drop_tracking_query_parameter(key)
    ]


class URLCollector(ABC):
    """A class to collect and de-duplicate URLs."""

    def _filter_tracking_query_entries(
        self,
        queries: List[Tuple[str, str]],
        remove_all: bool = False,
    ) -> List[Tuple[str, str]]:
        """Filter tracking query entries from an already parsed query list.

        Args:
            queries: Parsed query parameters as (key, value) pairs.
            remove_all: If True, return an empty list regardless of content.
        """
        return filter_tracking_query_entries(queries=queries, remove_all=remove_all)

    def _remove_tracking_parameters(self, url: str) -> str:
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
        filtered_queries = self._filter_tracking_query_entries(
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

    @abstractmethod
    async def add_previously_collected_urls(self, urls: List[str]) -> None:
        """Add a set of previously collected URLs to the internal state.

        Args:
            urls: A set of URLs that have been collected in previous runs.
        """
        pass

    @abstractmethod
    async def apply(self, product: ProductItem) -> ProductItem:
        """Manages the collection and deduplication of ProductItems.

        Args:
            product: The product item to process.
        """
        pass


class LocalURLCollector(URLCollector):
    """A class to collect and de-duplicate URLs using local storage."""

    def __init__(self):
        self._collected_currently: Set[str] = set()
        self._collected_previously: Set[str] = set()

    async def add_previously_collected_urls(self, urls: List[str]) -> None:
        """Add a set of previously collected URLs to the internal state.

        Args:
            urls: A set of URLs that have been collected in previous runs.
        """
        self._collected_previously.update(urls)

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
            product.filtered_at_stage = FilteredAtStage.URL_COLLECTION_CURRENT.value
            logger.debug(f"URL {url} already collected in current run")

        # deduplicate on previous runs coming from a db
        elif url in self._collected_previously:
            product.filtered = True
            product.filtered_at_stage = FilteredAtStage.URL_COLLECTION_PREVIOUS.value
            logger.debug(f"URL {url} as already collected in previous run")

        # Add to currently collected URLs
        else:
            self._collected_currently.add(url)

        return product


class DistributedURLCollector(URLCollector):
    """A URL collector that de-duplicates across pipeline runs using Redis.

    Seen URLs are stored under a namespaced Redis key computed as
    ``md5(cleaned_url + id_suffix)``. Hashing keeps keys fixed-length and
    bounded in memory regardless of URL length. An optional ``id_suffix``
    scopes deduplication beyond the Redis namespace (e.g. per-tenant,
    per-campaign) by changing the hash input.
    """

    def __init__(
        self,
        redis_config: RedisConfig,
        id_suffix: str = "",
    ) -> None:
        """Initialize the distributed collector.

        Args:
            redis_config: Redis configuration object.
            id_suffix: String appended to the URL before hashing to deduplication
        """
        self._id_suffix = id_suffix
        self._ttl = redis_config.ttl
        # Uses RedisBackend + default StringSerializer (RedisCacher uses
        # Cache+PickleSerializer): dedup markers are short strings and stay
        # inspectable in redis-cli.
        self._cache: RedisBackend = RedisBackend(
            endpoint=redis_config.hostname,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
            namespace=redis_config.namespace,
        )

    def _get_redis_key(self, url: str) -> str:
        """Return the MD5 hex digest of the `cleaned_url` concatenated with `id_suffix`.

        MD5 is used for deduplication only (not for security).

        Args:
            url: Tracking-parameter-free URL to hash.
        """
        payload = f"{url}{self._id_suffix}".encode("utf-8")
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()

    async def add_previously_collected_urls(self, urls: List[str]) -> None:
        """Seed Redis with already-seen URLs.

        Each URL is cleaned of tracking parameters, hashed and stored in
        Redis with TTL so subsequent ``apply()`` calls filter it out.

        Args:
            urls: URLs collected in previous runs.
        """
        for url in urls:
            key = self._get_redis_key(url)
            value = FilteredAtStage.URL_COLLECTION_PREVIOUS.value
            await self._cache.set(key=key, value=value, ttl=self._ttl)

    async def apply(self, product: ProductItem) -> ProductItem:
        """De-duplicate product cross-run using Redis.

        Args:
            product: The product item to process.
        """
        logger.debug(f'Processing de-duplication of product with url="{product.url}"')

        # Remove tracking parameters from the URL
        url = self._remove_tracking_parameters(product.url)
        product.url = url

        key = self._get_redis_key(url)
        value = await self._cache.get(key=key)

        # already seen in current run
        if value == FilteredAtStage.URL_COLLECTION_CURRENT.value:
            product.filtered = True
            product.filtered_at_stage = FilteredAtStage.URL_COLLECTION_CURRENT.value
            logger.debug(f"URL {url} already collected in current run")

        # already seen in a previous run (added via `add_previously_collected_urls`)
        elif value == FilteredAtStage.URL_COLLECTION_PREVIOUS.value:
            product.filtered = True
            product.filtered_at_stage = FilteredAtStage.URL_COLLECTION_PREVIOUS.value
            logger.debug(f"URL {url} already collected in previous run (distributed)")

        # first sighting -> mark as current
        elif value is None:
            logger.debug(f"Add url={url} to currently collected urls in redis")
            await self._cache.set(key=key, value=FilteredAtStage.URL_COLLECTION_CURRENT.value, ttl=self._ttl)
        
        else:
            raise ValueError(f"Redis returned value={value} for key={key} (url={url})")

        return product
