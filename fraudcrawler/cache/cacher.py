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
    REDIS_USE_CACHE,
    REDIS_URL,
    REDIS_TTL,
    REDIS_NAMESPACE,
)


logger = logging.getLogger(__name__)


class RedisCacher(ABC):
    """Abstract base class that adds Redis caching.

    :class:`RedisCacher` is used as a parent class for a subclass with an
    `apply()` method that should be wrapped inside a caching mechanism.

    Any subclass of RedisCacher must implement `apply()` with their core logic.
    The function `capply()` is a wrapper taking care of caching.

    Cache keys are built deterministically from the class name and serialized
    arguments (including Pydantic models), so identical calls produce
    identical results.
    """

    _default_host = "localhost"
    _default_port = 6379
    _default_db = 0

    def __init__(
        self,
        use_cache: bool = REDIS_USE_CACHE,
        url: str = REDIS_URL,
        ttl: int = REDIS_TTL,
        namespace: str = REDIS_NAMESPACE,
    ) -> None:
        """Initialize the cacher, optionally connecting to Redis.

        Args:
            use_cache: Whether to use Redis cache.
            url: Redis connection URL (redis:// or rediss://).
            ttl: Time-to-live in seconds for cached entries.
            namespace: Key namespace to isolate entries in shared Redis instances.
        """
        # Input parameters
        self._use_cache = use_cache
        self._ttl = ttl
        self._namespace = namespace

        # Parameters for caching
        self._cache: RedisCache | None = None
        if self._use_cache:
            redis_kwargs = self._get_redis_kwargs(url=url)
            self._cache = cast(
                RedisCache,
                Cache(
                    cache_class=Cache.REDIS,  # type: ignore[reportArgumentType]
                    serializer=PickleSerializer(),
                    namespace=self._namespace,
                    **redis_kwargs,
                ),
            )

    def _get_redis_kwargs(self, url: str) -> Dict[str, str | int | None]:
        """Get redis parameters as endpoint, port, password and db"""

        # Parse and check url
        u = urlparse(url)
        if u.scheme not in {"redis", "rediss"}:
            raise ValueError("redis_url must start with redis:// or rediss://")

        # Create and return redis kwargs
        return {
            "endpoint": u.hostname or self._default_host,
            "port": u.port or self._default_port,
            "password": u.password,
            "db": int(up) if (up := u.path.lstrip("/")) else self._default_db,
        }

    @staticmethod
    def _stable_key(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))

    @staticmethod
    def _serialize_object(obj: Any) -> Any:
        """Recursively serialize args/kwargs for cache keys.

        Uses model_dump() for pydantic.BaseModels, recurses into list and dict,
        leaves the rest unchanged.
        """
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, Sequence):
            return [RedisCacher._serialize_object(x) for x in obj]
        if isinstance(obj, Dict):
            return {k: RedisCacher._serialize_object(v) for k, v in obj.items()}
        return obj

    def _build_key(self, *args: Any, **kwargs: Any) -> str:
        """Builds caching key based on class name, args and kwargs."""
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
        """Cached wrapper around self.apply() method."""

        # Check if self._cache has been defined
        if self._cache is None:
            raise RuntimeError("Redis cache not initialized")

        # Get caching key from arguments
        key = self._build_key(*args, **kwargs)

        # Check if key exists in the cacher; otherwise compute the response
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
        """The cached function that each child of :class:`RedisCacher` must implement."""
        pass

    async def capply(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the method `apply()` with Redis caching if enabled. Otherwise it calls `apply()` directly."""

        # Cacher wrapped around self.apply() method
        if self._use_cache:
            logger.debug(f"Running cached apply() for {self.__class__.__name__}")
            result = await self._cached_apply(*args, **kwargs)

        # No cacher, simply run self.apply() method
        else:
            logger.debug(f"Running not-cached apply() for {self.__class__.__name__}")
            result = await self.apply(*args, **kwargs)

        return result

    # ---------------------------------
    # Utils for managing Redis remotely
    # ---------------------------------
    async def utils_clear_namespace(self) -> None:
        if self._use_cache:
            if self._cache is None:
                raise RuntimeError("Redis cache not initialized")
            await self._cache.clear()

    async def utils_invalidate(self, *args: Any, **kwargs: Any) -> None:
        if self._use_cache:
            if self._cache is None:
                raise RuntimeError("Redis cache not initialized")
            await self._cache.delete(key=self._build_key(*args, **kwargs))

    async def utils_redis_is_available(self) -> bool:
        """Works with aiocache backends: does a small SET/GET/DEL roundtrip."""
        # Dummy key-value pair
        key = f"__healthcheck__:{self.__class__.__name__}:{uuid.uuid4().hex}"
        value = "1337"
        test_ttl = 5

        # Check if cache is defined at all
        if self._cache is None:
            raise RuntimeError("Redis cache not initialized")

        # Try to set dummy key-value pair
        try:
            logger.debug("test to set dummy key-value pair in cacher")
            await self._cache.set(key=key, value=value, ttl=test_ttl)
        except Exception:
            logger.error("failed to set dummy key-value pair in cacher", exc_info=True)
            return False

        # Try to read dummy key-value pair and compare it
        try:
            logger.debug("read written dummy value")
            obtained = await self._cache.get(key=key)
            await self._cache.delete(key=key)
            return obtained == value
        except Exception:
            logger.error("failed to read dummy key-value pair in cacher", exc_info=True)
            return False
