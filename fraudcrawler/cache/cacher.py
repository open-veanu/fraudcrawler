from abc import ABC, abstractmethod
import json
import logging
from pydantic import BaseModel
from typing import Any, cast, Dict, Sequence
from urllib.parse import urlparse
import uuid

from aiocache import Cache
from aiocache.backends.redis import RedisCache
from aiocache.serializers import PickleSerializer

from fraudcrawler.settings import (
    REDIS_CACHE_NAMESPACE,
    REDIS_DEFAULT_URL,
    REDIS_TTL,
    REDIS_USE_CACHE,
)


logger = logging.getLogger(__name__)


_DEFAULT_REDIS_HOST = "localhost"
_DEFAULT_REDIS_PORT = 6379
_DEFAULT_REDIS_DB = 0


def parse_redis_url(url: str) -> Dict[str, str | int | None]:
    """Parse a redis:// or rediss:// URL into aiocache connection kwargs.

    Args:
        url: Redis connection URL (redis:// or rediss://).

    Returns:
        Dict with keys ``endpoint`` (str), ``port`` (int), ``password``
        (str | None), and ``db`` (int), suitable for passing as kwargs to
        aiocache Redis backends.

    Raises:
        ValueError: If ``url`` does not start with ``redis://`` or ``rediss://``.
    """
    u = urlparse(url)
    if u.scheme not in {"redis", "rediss"}:
        raise ValueError("redis_url must start with redis:// or rediss://")
    return {
        "endpoint": u.hostname or _DEFAULT_REDIS_HOST,
        "port": u.port or _DEFAULT_REDIS_PORT,
        "password": u.password,
        "db": int(up) if (up := u.path.lstrip("/")) else _DEFAULT_REDIS_DB,
    }


class RedisCacher(ABC):
    """Abstract base class that adds Redis caching to a subclass.

    RedisCacher wraps a subclass's apply() method with a transparent cache
    layer. Subclasses implement apply() with their core logic and call
    capply() as the public entry point, which handles cache lookup, result
    storage, and bypassing Redis when use_cache is False.

    Cache keys are built deterministically from the class name and the
    serialized call arguments (including Pydantic models), so identical
    calls always map to the same cache entry.
    """

    def __init__(
        self,
        use_cache: bool = REDIS_USE_CACHE,
        url: str = REDIS_DEFAULT_URL,
        ttl: int = REDIS_TTL,
        namespace: str = REDIS_CACHE_NAMESPACE,
    ) -> None:
        """Initialize the cacher, optionally connecting to Redis.

        Args:
            use_cache: Whether to use Redis cache.
            url: Redis connection URL (redis:// or rediss://).
            ttl: Time-to-live in seconds for cached entries.
            namespace: Key namespace to isolate entries in shared Redis instances.
        """
        self._use_cache = use_cache
        self._ttl = ttl
        self._namespace = namespace

        self._cache: RedisCache | None = None
        if self._use_cache:
            redis_kwargs = parse_redis_url(url=url)
            self._cache = cast(
                RedisCache,
                Cache(
                    cache_class=Cache.REDIS,  # type: ignore[reportArgumentType]
                    serializer=PickleSerializer(),
                    namespace=self._namespace,
                    **redis_kwargs,
                ),
            )

    @staticmethod
    def _stable_key(payload: Dict[str, Any]) -> str:
        """Serialize a payload dict to a compact, deterministic JSON string.

        Args:
            payload: Dict to serialize as a cache key.
        """
        return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))

    @staticmethod
    def _serialize_object(obj: Any) -> Any:
        """Recursively convert a value to a JSON-serializable representation.

        Converts pydantic BaseModel instances via model_dump(), recurses into
        lists and dicts, and returns all other values unchanged.

        Args:
            obj: Value to serialize.
        """
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, str):
            return obj
        if isinstance(obj, Sequence):
            return [RedisCacher._serialize_object(x) for x in obj]
        if isinstance(obj, Dict):
            return {k: RedisCacher._serialize_object(v) for k, v in obj.items()}
        return obj

    def _build_key(self, *args: Any, **kwargs: Any) -> str:
        """Build a deterministic cache key from the class name and call arguments.

        Args:
            *args: Positional arguments passed to apply().
            **kwargs: Keyword arguments passed to apply().
        """
        args_ = tuple(self._serialize_object(a) for a in args)
        kwargs_ = {k: self._serialize_object(v) for k, v in kwargs.items()}
        return self._stable_key(
            {
                "cls": self.__class__.__name__,
                "args": args_,
                "kwargs": kwargs_,
            }
        )

    async def _cached_apply(self, *args: Any, **kwargs: Any) -> Any:
        """Execute apply() with a Redis cache lookup.

        Returns the cached result on a hit; otherwise calls apply(), stores
        the result, and returns it.

        Args:
            *args: Positional arguments forwarded to apply().
            **kwargs: Keyword arguments forwarded to apply().

        Returns:
            Result of apply(), either retrieved from cache or freshly computed.

        Raises:
            RuntimeError: If the Redis cache has not been initialized.
        """
        if self._cache is None:
            raise RuntimeError("Redis cache not initialized")

        key = self._build_key(*args, **kwargs)

        exists = await self._cache.exists(key=key)
        if exists:
            logger.debug(
                f"Found cached response for {self.__class__.__name__}.apply(args={args}, kwargs={kwargs})"
            )
            result = await self._cache.get(key=key)
        else:
            logger.debug(
                f"No cached response for {self.__class__.__name__}.apply(args={args}, kwargs={kwargs})"
            )
            result = await self.apply(*args, **kwargs)

            logger.debug(
                f"Set cached response for {self.__class__.__name__}.apply(args={args}, kwargs={kwargs})"
            )
            await self._cache.set(key=key, value=result, ttl=self._ttl)

        return result

    @abstractmethod
    async def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Core logic that each subclass must implement.

        Called by capply() and, when caching is enabled, by _cached_apply().
        Subclasses should not call this method directly; use capply() instead.

        Args:
            *args: Positional arguments specific to the subclass.
            **kwargs: Keyword arguments specific to the subclass.

        Returns:
            Subclass-defined result that will be cached under the call key.
        """
        pass

    async def capply(self, *args: Any, **kwargs: Any) -> Any:
        """Call apply() with Redis caching when enabled, or directly otherwise.

        Args:
            *args: Positional arguments forwarded to apply().
            **kwargs: Keyword arguments forwarded to apply().

        Returns:
            Result of apply(), either retrieved from cache or freshly computed.
        """
        if self._use_cache:
            logger.debug(f"Running cached apply() for {self.__class__.__name__}")
            result = await self._cached_apply(*args, **kwargs)
        else:
            logger.debug(f"Running not-cached apply() for {self.__class__.__name__}")
            result = await self.apply(*args, **kwargs)

        return result

    # ---------------------------------
    # Utils for managing Redis remotely
    # ---------------------------------

    async def utils_clear_namespace(self) -> None:
        """Delete all cache entries in this instance's namespace.

        No-op when caching is disabled.

        Raises:
            RuntimeError: If caching is enabled but the cache is not initialized.
        """
        if self._use_cache:
            if self._cache is None:
                raise RuntimeError("Redis cache not initialized")
            await self._cache.clear()

    async def utils_invalidate(self, *args: Any, **kwargs: Any) -> None:
        """Delete the cache entry for a specific set of call arguments.

        No-op when caching is disabled.

        Args:
            *args: Positional arguments identifying the cache entry to remove.
            **kwargs: Keyword arguments identifying the cache entry to remove.

        Raises:
            RuntimeError: If caching is enabled but the cache is not initialized.
        """
        if self._use_cache:
            if self._cache is None:
                raise RuntimeError("Redis cache not initialized")
            await self._cache.delete(key=self._build_key(*args, **kwargs))

    async def utils_redis_is_available(self) -> bool:
        """Check Redis availability with a SET/GET/DEL health-check roundtrip.

        Raises:
            RuntimeError: If the Redis cache has not been initialized.
        """
        key = f"__healthcheck__:{self.__class__.__name__}:{uuid.uuid4().hex}"
        value = "1337"
        test_ttl = 5

        if self._cache is None:
            raise RuntimeError("Redis cache not initialized")

        try:
            logger.debug("test to set dummy key-value pair in cacher")
            await self._cache.set(key=key, value=value, ttl=test_ttl)
        except Exception:
            logger.error("failed to set dummy key-value pair in cacher", exc_info=True)
            return False

        try:
            logger.debug("read written dummy value")
            obtained = await self._cache.get(key=key)
            await self._cache.delete(key=key)
            return obtained == value
        except Exception:
            logger.error("failed to read dummy key-value pair in cacher", exc_info=True)
            return False
