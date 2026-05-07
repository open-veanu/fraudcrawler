from abc import ABC, abstractmethod
import hashlib
import logging
from typing import List, Set, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, quote, urlunparse, ParseResult

from aiocache.backends.redis import RedisBackend
from tenacity import RetryCallState

from fraudcrawler.base.base import DomainUtils, FilteredAtStage, ProductItem
from fraudcrawler.base.retry import get_async_retry
from fraudcrawler.cache.cacher import RedisConfig
from fraudcrawler.settings import (
    KNOWN_TRACKERS,
    REDIS_CONNECTION_TIMEOUT,
    REDIS_MULTI_SET_BATCH_SIZE,
)

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


class DistributedURLCollector(URLCollector, DomainUtils):
    """A URL collector that de-duplicates across pipeline runs using Redis.

    Seen URLs are stored under a namespaced Redis key of the form
    ``{domain}_{sha256(cleaned_url + id_suffix)}``. The domain prefix keeps
    keys inspectable in redis-cli; the hash bounds key length regardless of
    URL size. An optional ``id_suffix`` scopes deduplication beyond the
    Redis namespace (e.g. per-tenant, per-campaign) by changing the hash
    input.
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

        # Uses RedisBackend + default StringSerializer
        self._cache: RedisBackend = RedisBackend(
            endpoint=redis_config.hostname,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
            namespace=redis_config.namespace,
            timeout=REDIS_CONNECTION_TIMEOUT,
        )

    @classmethod
    def _log_cache_before(cls, op: str, key: str, retry_state: RetryCallState) -> None:
        """Context aware logging before a Redis attempt."""
        if retry_state.attempt_number > 1:
            logger.debug(
                f"Retry attempt {retry_state.attempt_number} of "
                f"{cls.__name__} cache {op}(key={key})."
            )

    @classmethod
    def _log_cache_before_sleep(
        cls, op: str, key: str, retry_state: RetryCallState
    ) -> None:
        """Context aware logging before sleeping after a failed Redis attempt."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f"Attempt {retry_state.attempt_number} of {cls.__name__} cache "
                f"{op}(key={key}) failed with error: {retry_state.outcome.exception()}. "
                f"Retrying in {retry_state.upcoming_sleep:.0f} seconds."
            )

    def _get_redis_key(self, url: str) -> str:
        """Return a Redis key of the form ``{domain}_{sha256_hex}``.

        Note:
          - The SHA-256 digest is computed over the URL concatenated with
            ``id_suffix``. The domain prefix (via ``DomainUtils._get_domain``)
            keeps keys inspectable in redis-cli.
          - As here we are using RedisBackend, we need to manually put a ":"
            in front of the key to split it in folders

        Args:
            url: Tracking-parameter-free URL to hash.
        """
        payload = f"{url}{self._id_suffix}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        domain = self._get_domain(url)
        return f":{domain}_{digest}"

    async def _add_url(self, key: str, value: str) -> None:
        """Set a URL marker in Redis under `get_async_retry`.

        Args:
            key: Redis key (already namespaced/hashed via `_get_redis_key`).
            value: One of ``FilteredAtStage.URL_COLLECTION_*`` markers.
        """
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_cache_before(
            op="set", key=key, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_cache_before_sleep(
            op="set", key=key, retry_state=retry_state
        )
        try:
            async for attempt in retry:
                with attempt:
                    await self._cache.set(key=key, value=value, ttl=self._ttl)
        except Exception:
            logger.warning(
                f"Cache set(key={key}) failed after retries. "
                f"URL marker not stored; URL may be re-crawled.",
                exc_info=True,
            )

    async def add_previously_collected_urls(self, urls: List[str]) -> None:
        """Seed Redis with already-seen URL.

        Args:
            urls: URLs collected in previous runs.
        """
        if not urls:
            return

        logger.debug(f"adding {len(urls)} previously collected urls")

        pairs = [
            (self._get_redis_key(url), FilteredAtStage.URL_COLLECTION_PREVIOUS.value)
            for url in urls
        ]

        for start in range(0, len(pairs), REDIS_MULTI_SET_BATCH_SIZE):
            end = start + REDIS_MULTI_SET_BATCH_SIZE
            logger.debug(
                f"adding batch of previously collected urls (start={start}; end={end})"
            )
            chunk = pairs[start:end]
            log_key = f"<bulk:{len(chunk)}@{start}>"

            retry = get_async_retry()
            retry.before = lambda retry_state, _lk=log_key: self._log_cache_before(  # type: ignore[misc]
                op="multi_set", key=_lk, retry_state=retry_state
            )
            retry.before_sleep = (
                lambda retry_state, _lk=log_key: self._log_cache_before_sleep(  # type: ignore[misc]
                    op="multi_set", key=_lk, retry_state=retry_state
                )
            )
            try:
                async for attempt in retry:
                    with attempt:
                        await self._cache.multi_set(pairs=chunk, ttl=self._ttl)
            except Exception:
                logger.warning(
                    f"Cache multi_set chunk ({len(chunk)} URLs @ offset {start}) "
                    f"failed after retries. Markers in this chunk not stored.",
                    exc_info=True,
                )

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

        # Cache lookup with retry; fall back to "first sighting" on exhaustion.
        #  - `before`: before the request is made (and before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_cache_before(
            op="get", key=key, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_cache_before_sleep(
            op="get", key=key, retry_state=retry_state
        )
        value: str | None = None
        try:
            async for attempt in retry:
                with attempt:
                    value = await self._cache.get(key=key)
        except Exception:
            logger.warning(
                f"Cache get(key={key}) failed after retries. "
                f"Falling back to live collection.",
                exc_info=True,
            )
            value = None

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

        # first sighting (genuine miss OR retry-exhausted) -> mark as current
        elif value is None:
            logger.debug(f"Add url={url} to currently collected urls in redis")
            await self._add_url(
                key=key, value=FilteredAtStage.URL_COLLECTION_CURRENT.value
            )

        else:
            raise ValueError(f"Redis returned value={value} for key={key} (url={url})")

        return product
