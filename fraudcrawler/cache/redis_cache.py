import asyncio
import hashlib
import json
import logging
import os
from typing import Awaitable, Callable, Mapping, Optional, TypeVar, cast

from aiocache import Cache
from aiocache.lock import RedLock
from aiocache.serializers import PickleSerializer
import redis.asyncio as redis
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

R = TypeVar("R")


class RedisCacher:
    """Components can inherit from this class to get access to the `capply()` method
    for caching any async method call. Cache behavior is configured via `__init__`
    parameters with defaults from environment variables.
    """

    def __init__(
        self,
        use_cache: bool | None = None,
        redis_url: str | None = None,
        redis_cache_ttl: int | None = None,
        lock_lease: int = 60,
        wait_timeout: float = 30.0,
        poll_interval: float = 0.2,
    ):
        """Initialize RedisCacher with cache configuration.

        Args:
            use_cache: Whether to enable caching. Defaults to REDIS_USE_CACHE env var or True.
            redis_url: Redis connection URL. Defaults to REDIS_URL env var or localhost.
            redis_cache_ttl: Time-to-live for cache entries in seconds. Defaults to REDIS_CACHE_TTL env var or 86400.
            lock_lease: Lock lease time in seconds. Defaults to 60.
            wait_timeout: Maximum time to wait for cache in seconds. Defaults to 30.0.
            poll_interval: Interval between cache polls in seconds. Defaults to 0.2.
        """
        if use_cache is None:
            use_cache_str = os.getenv("REDIS_USE_CACHE", "true")
            self._use_cache = use_cache_str.lower() in ("true", "1", "yes", "on")
        else:
            self._use_cache = use_cache

        self._url = redis_url or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        self._ttl = redis_cache_ttl or int(os.getenv("REDIS_CACHE_TTL", "86400"))
        # Validate parameters
        if lock_lease <= 0:
            raise ValueError("lock_lease must be positive")
        if wait_timeout <= 0:
            raise ValueError("wait_timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        self._lock_lease = lock_lease
        self._wait_timeout = wait_timeout
        self._poll_interval = poll_interval

        # Initialize cache instance if caching is enabled
        self._cache: Optional[Cache] = None
        self._redis_unavailable = False  # Track if we've detected Redis is unavailable
        if self._use_cache:
            self._cache = self._create_cache(self._url)
            if self._cache is None:
                self._use_cache = False
                self._redis_unavailable = True

    @staticmethod
    def _create_cache(redis_url: str) -> Optional[Cache]:
        """Create a Redis cache instance from a Redis URL."""
        try:
            # Parse Redis URL to extract connection parameters
            parsed = urlparse(redis_url)
            endpoint = parsed.hostname or "localhost"
            port = parsed.port or 6379
            password = parsed.password
            db = (
                int(parsed.path.lstrip("/"))
                if parsed.path and parsed.path != "/"
                else 0
            )

            # Create cache with RedisCache backend and explicit timeout configuration
            cache_kwargs = {
                "endpoint": endpoint,
                "port": port,
                "db": db,
                "namespace": "resp_cache",
            }
            # Only include password if provided in URL
            if password:
                cache_kwargs["password"] = password

            cache = Cache(Cache.REDIS, **cache_kwargs)
            cache.serializer = PickleSerializer()
            logger.info(
                f"Initialized Redis cache: {redis_url} (namespace={cache.namespace})"
            )
            return cache
        except Exception as e:
            logger.warning(
                f"Failed to initialize Redis cache: {e}. Caching will be disabled."
            )
            return None

    async def capply(
        self,
        key_builder: Callable[..., Mapping[str, object]],
        *args,
        func: Callable[..., Awaitable[R]] | None = None,
        **kwargs,
    ) -> R:
        """Cache a call to self.apply() or a provided function.

        Args:
            key_builder: Required function that builds the cache key from args/kwargs.
            *args: Positional arguments to pass to the function.
            func: Function to cache. If None, uses self.apply().
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Result from the function, potentially from cache.
        """
        target_func = func if func is not None else self.apply

        if not self._use_cache:
            return await target_func(*args, **kwargs)

        async def call_uncached() -> R:
            return await target_func(*args, **kwargs)

        # Build signature payload using key_builder
        try:
            signature_payload = key_builder(*args, **kwargs)
        except Exception:
            logger.exception(
                f"key_builder failed for {self.__class__.__name__}.apply; "
                "bypassing cache"
            )
            return await call_uncached()

        if not isinstance(signature_payload, Mapping):
            logger.warning(
                f"Invalid signature_payload for {self.__class__.__name__}.apply; "
                "bypassing cache"
            )
            return await call_uncached()

        # Build cache key with class name namespace
        try:
            cache_key = self._build_cache_key(signature_payload)
        except Exception:
            logger.exception(
                f"Cache key build failed for {self.__class__.__name__}.apply; "
                "bypassing cache"
            )
            return await call_uncached()

        if self._cache is None:
            return await call_uncached()

        ctx = self._log_context(signature_payload)

        # Try to get from cache
        try:
            cached_value = await self._cache_get_or_purge(cache_key)
        except Exception as e:
            if not self._redis_unavailable:
                # First time detecting Redis is unavailable - log prominently
                logger.error(
                    f"Redis connection failed for {self.__class__.__name__}. "
                    f"Cannot connect to {self._url}. "
                    f"All cache operations will be bypassed. Error: {e}"
                )
                self._redis_unavailable = True
            logger.warning(
                f"Cache get failed for {self.__class__.__name__}.apply; "
                f"bypassing cache ({ctx})"
            )
            return await call_uncached()

        if cached_value is not None:
            logger.info(f"Using cache for {ctx}")
            return cast(R, cached_value)

        logger.info(f"Cache miss for {ctx}")

        # Lock + compute path
        lock_key = f"{cache_key}:lock"
        try:
            async with RedLock(self._cache, lock_key, lease=self._lock_lease):
                # Re-check after acquiring lock
                cached_value = await self._cache_get_or_purge(cache_key)
                if cached_value is not None:
                    logger.info(f"Using cache (LOCK-RACE-HIT) for {ctx}")
                    return cast(R, cached_value)

                result = await call_uncached()

                try:
                    await self._cache.set(cache_key, result, ttl=self._ttl)
                    logger.info(f"Computed and cached result for {ctx}")
                except Exception:
                    logger.exception(
                        f"Cache set failed for {self.__class__.__name__}.apply "
                        f"(continuing) ({ctx})"
                    )

                return result
        except Exception:
            logger.warning(
                f"Lock path failed for {self.__class__.__name__}.apply; "
                f"waiting up to {self._wait_timeout:.1f}s ({ctx})"
            )

        # Wait/poll path
        v = await self._wait_for_cache(cache_key)
        if v is not None:
            logger.info(f"Using cache (WAIT-HIT) for {ctx}")
            return cast(R, v)

        # Fallback compute
        logger.info(f"Cache unavailable (timeout); computing without cache for {ctx}")
        result = await call_uncached()
        try:
            await self._cache.set(cache_key, result, ttl=self._ttl)
            logger.info(f"Computed and cached result (post-timeout) for {ctx}")
        except Exception as exc:
            logger.debug(
                f"Post-timeout cache set failed for {ctx} (ignored): {exc}",
                exc_info=True,
            )
        return result

    async def apply(self, *args, **kwargs):
        """The 'Implementation'. Each child should define this, or use capply() with func= parameter.

        This method can be overridden by subclasses. If not overridden and capply() is called
        without a func= parameter, a NotImplementedError will be raised.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.apply() not implemented. "
            "Either implement apply() or use capply() with func= parameter."
        )

    def _build_cache_key(self, signature_payload: Mapping[str, object]) -> str:
        """Build a cache key from signature payload, namespaced by class name."""
        # Exclude logging-only fields from cache key
        payload_for_key = {k: v for k, v in signature_payload.items() if k != "url"}
        serialized = json.dumps(
            payload_for_key, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        payload_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        # Namespace by class name and include redis_url identifier to avoid collisions
        class_name = self.__class__.__name__
        # Use a hash of redis_url to keep key reasonable length
        redis_id = hashlib.sha256(self._url.encode("utf-8")).hexdigest()[:8]
        return f"{class_name}:{redis_id}:{payload_hash}"

    async def test_redis_is_running(self, timeout: float = 2.0) -> bool:
        """Test if Redis connection is available.
        """
        if not self._use_cache or self._cache is None:
            return False

        # Use redis-py as aiocache does not provide a way to test if Redis is running
        test_client = None
        try:
            test_client = redis.from_url(
                self._url,
                socket_connect_timeout=timeout,
                socket_timeout=timeout,
                decode_responses=False,  # aiocache uses decode_responses=False
                single_connection_client=True,  # Force single connection for testing
            )

            # Try to ping Redis with timeout - this will force a connection attempt
            ping_result = await asyncio.wait_for(test_client.ping(), timeout=timeout)

            if ping_result:
                self._redis_unavailable = False
                return True
            return False
        except asyncio.TimeoutError:
            self._handle_connection_error(
                f"Redis connection test timed out for {self.__class__.__name__}. "
                f"Cannot connect to {self._url} within {timeout}s. "
                "Redis appears to be unavailable."
            )
            return False
        except (
            redis.ConnectionError,
            redis.TimeoutError,
            OSError,
            ConnectionRefusedError,
        ) as e:
            # These are expected when Redis is down
            self._handle_connection_error(
                f"Redis connection test failed for {self.__class__.__name__}. "
                f"Cannot connect to {self._url}. Error: {e}"
            )
            return False
        except Exception as e:
            # Catch any other unexpected errors
            self._handle_connection_error(
                f"Redis connection test failed for {self.__class__.__name__}. "
                f"Cannot connect to {self._url}. Error: {e}"
            )
            return False
        finally:
            # Clean up the test client
            if test_client is not None:
                try:
                    await test_client.aclose()
                except Exception as exc:
                    logger.debug(
                        f"Error closing test Redis client (ignored): {exc}",
                        exc_info=True,
                    )

    def _handle_connection_error(self, message: str) -> None:
        """Handle Redis connection error by logging and updating availability status."""
        if not self._redis_unavailable:
            logger.error(message)
        self._redis_unavailable = True

    @staticmethod
    def _is_incompatible_cache_entry_error(e: Exception) -> bool:
        """Check if exception indicates incompatible cache entry."""
        msg = str(e)
        name = type(e).__name__
        return ("UnpicklingError" in name) or ("invalid load key" in msg)

    async def _cache_get_or_purge(self, key: str) -> object | None:
        """Get cached value, purging if incompatible.

        Returns cached value if present. If entry exists but can't be deserialized
        due to serializer mismatch, delete and treat as miss.

        Returns:
            Cached value or None.
        """
        if self._cache is None:
            return None

        try:
            return await self._cache.get(key)
        except Exception as e:
            if self._is_incompatible_cache_entry_error(e):
                try:
                    await self._cache.delete(key)
                except Exception as exc:
                    logger.debug(
                        f"Failed to purge incompatible cache entry key={key!r} (ignored): {exc}",
                        exc_info=True,
                    )
                return None
            raise

    async def _wait_for_cache(self, key: str) -> object | None:
        """Wait for cache entry to appear, polling at intervals."""
        if self._cache is None:
            return None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        deadline = loop.time() + self._wait_timeout
        while loop.time() < deadline:
            await asyncio.sleep(self._poll_interval)
            try:
                v = await self._cache_get_or_purge(key)
                if v is not None:
                    return v
            except Exception as exc:
                logger.debug(
                    f"Transient cache issue during wait for key={key!r} (ignored): {exc}",
                    exc_info=True,
                )
        return None

    @staticmethod
    def _log_context(signature_payload: Mapping[str, object]) -> str:
        """Build a log context string from signature payload."""
        endpoint = str(signature_payload.get("endpoint", "?"))
        provider = signature_payload.get("provider")

        # serpapi has no url parameter, so we use the search query
        if provider == "serpapi" and (q := signature_payload.get("q")):
            url = f"q={q}"
        else:
            url_val = signature_payload.get("url") or signature_payload.get(
                "request_url"
            )
            url = str(url_val) if url_val is not None else "?"

        if provider:
            return f"provider={provider} endpoint={endpoint} url={url}"
        return f"endpoint={endpoint} url={url}"


if __name__ == "__main__":

    async def main():
        print("Testing Redis connection...")
        cacher = RedisCacher()
        is_available = await cacher.test_redis_is_running()
        if is_available:
            print("Redis connection test: Available")
        else:
            print("Redis connection test: Unavailable (caching will be bypassed)")

    asyncio.run(main())
