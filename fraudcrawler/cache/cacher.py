from abc import ABC, abstractmethod
import json
import logging
from pydantic import BaseModel
from typing import Any, Callable, cast, Dict, List
from urllib.parse import urlparse
import uuid

from aiocache import Cache
from aiocache.backends.redis import RedisCache
from aiocache.serializers import JsonSerializer

from fraudcrawler.settings import (
    REDIS_USE_CACHE,
    REDIS_URL,
    REDIS_TTL,
    REDIS_NAMESPACE,
    REDIS_LEASE,
)


logger = logging.getLogger(__name__)


class _PydanticJsonSerializer(JsonSerializer):
    """JsonSerializer that converts Pydantic models via model_dump() before JSON encoding."""

    @staticmethod
    def _pydantic_json_default(o: Any) -> Any:
        if hasattr(o, "model_dump"):
            return o.model_dump()
        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable"
        )

    def dumps(self, value: Any) -> str:
        return json.dumps(value, default=self._pydantic_json_default)


class RedisCacher(ABC):
    """Abstract base class that adds Redis caching.

    Any subclass of RedisCacher must implement `apply()` with their core logic.
    The function `capply()` is a wrapper taking care of caching.

    Cache keys are built deterministically from the class name, method name, and
    serialized arguments (including Pydantic models), so identical calls produce
    identical results.
    """

    _default_host = 'localhost'
    _default_port = 6379
    _default_db = '0'

    def __init__(
        self,
        use_cache: bool = REDIS_USE_CACHE,
        url: str = REDIS_URL,
        ttl: int = REDIS_TTL,
        namespace: str = REDIS_NAMESPACE,
        lease: int = REDIS_LEASE,
    ) -> None:
        """Initialize the cacher, optionally connecting to Redis.

        Args:
            use_cache: Whether to use Redis cache.
            url: Redis connection URL (redis:// or rediss://).
            ttl: Time-to-live in seconds for cached entries.
            namespace: Key namespace to isolate entries in shared Redis instances.
            lease: Lease timeout in seconds for distributed locking.
        """
        # Input parameteras
        self._use_cache = use_cache
        self._ttl = ttl
        self._namespace = namespace
        self._lease = lease

        # Parameters for caching
        self._cache: RedisCache | None = None
        self._serializer = _PydanticJsonSerializer()
        if self._use_cache:
            redis_kwargs = self._get_redis_kwargs(url=url)
            self._cache = cast(RedisCache, Cache(
                cache_class=Cache.REDIS,    # type: ignore[reportArgumentType]
                serializer=self._serializer,
                namespace=self._namespace,
                **redis_kwargs,
            ))

    async def _redis_is_available(self) -> bool:
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
            logger.debug('test to set dummy key-value pair in cacher')
            await self._cache.set(key=key, value=value, ttl=test_ttl)
        except Exception:
            logger.error("failed to set dummy key-value pair in cacher", exc_info=True)
            return False
        
        # Try to read dummy key-value pair and compare it
        try:
            logger.debug(f'read writtine dummy value')
            obtained = await self._cache.get(key=key)
            await self._cache.delete(key=key)
            return obtained == value
        except Exception:
            logger.error('failed to read dummy key-value pair in cacher', exc_info=True)
            return False

    def _get_redis_kwargs(self, url: str) -> Dict[str, str | int | None]:
        """Get redis paramters as endpoint, port, password and db"""

        # Parse and check url
        u = urlparse(url)
        if u.scheme not in {"redis", "rediss"}:
            raise ValueError("redis_url must start with redis:// or rediss://")
        
        # Create and return redis kwargs
        return {
            "endpoint": u.hostname or self._default_host,
            "port": u.port or self._default_port,
            "password": u.password,
            "db": int((u.path or "/0").lstrip("/") or self._default_db),
        }

    @staticmethod
    def _stable_key(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))

    @staticmethod
    def _serialize_object(obj: Any) -> Any:
        """Recursively serialize args/kwargs for cache keys.
        
        Uses model_dump() for pydantic.BaseModels, recurses into list and dict,
        leaves the rest unchanged.
        """
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, List):
            return [RedisCacher._serialize_object(x) for x in obj]
        if isinstance(obj, dict):
            return {k: RedisCacher._serialize_object(v) for k, v in obj.items()}
        return obj

    def _build_key(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        """Builds caching key based on function name, args and kwargs."""
        args_ = tuple(self._serialize_object(a) for a in args)
        kwargs_ = {k: self._serialize_object(v) for k, v in kwargs.items()}
        return self._stable_key(
            {
                "cls": self.__class__.__name__,
                "fn": func.__name__,
                "args": args_,
                "kwargs": kwargs_,
            }
        )

    @staticmethod
    def _cache_log_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """Extract a short context (url, product.url, or search_term) for cache hit/miss logs."""
        # product with .url (e.g. OpenAIClassification)
        p = kwargs.get("product") or (
            args[0] if args and hasattr(args[0], "url") else None
        )
        if p is not None:
            u = getattr(p, "url", None)
            if isinstance(u, str) and u:
                return u
        # url (e.g. ZyteAPI)
        u = kwargs.get("url")
        if isinstance(u, str) and u:
            return u
        if (
            args
            and isinstance(args[0], str)
            and args[0].startswith(("http://", "https://"))
        ):
            return args[0]
        # search_term (e.g. Searcher, Enricher)
        st = kwargs.get("search_term")
        if isinstance(st, str) and st:
            return st
        return ""

    async def _cached_apply(self, *args: Any, **kwargs: Any) -> Any:
        """Cached wrapper around self.apply() method."""

        # Check if self._cache has been defined
        if self._cache is None:
            raise RuntimeError("Redis cache not initialized")
    
        # Get caching key from arguments
        key = self._build_key(self.apply, *args, **kwargs)

        # Check if key exists in the cacher
        val = await self._cache.get(key=key)
        if val is not None:
            logger.info("Cache hit [%s]%s", self.__class__.__name__, suffix)
            return val


        ctx = self._cache_log_context(args=args, kwargs=kwargs)
        if len(ctx) > 72:
            ctx = ctx[:69] + "..."
        suffix = f" | {ctx}" if ctx else ""
        logger.info("Cache miss [%s]%s", self.__class__.__name__, suffix)
        result = await self.apply(*args, **kwargs)
        await self._cache.set(key=key, value=result, ttl=self._ttl)
        return result

    async def capply(self, *args: Any, **kwargs: Any) -> Any:
        """(Cached) apply method."""
        # Cacher wrapped around self.apply() method
        if self._use_cache:
            logger.debug(f"Running cached apply() for {self.__class__.__name__}")
            response = await self._cached_apply(*args, **kwargs)
        
        # No cacher, simply run self.apply() method
        else:
            logger.debug(f"Running not-cached apply() for {self.__class__.__name__}")
            response = await self.apply(*args, **kwargs)
        
        return response

    async def invalidate(self, *args: Any, **kwargs: Any) -> None:
        if self._use_cache:
            if self._cache is None:
                raise RuntimeError("Redis cache not initialized")
            await self._cache.delete(key=self._build_key(self.apply, *args, **kwargs))

    async def clear_namespace(self) -> None:
        if self._use_cache:
            if self._cache is None:
                raise RuntimeError("Redis cache not initialized")
            await self._cache.clear()

    @abstractmethod
    async def apply(self, *args: Any, **kwargs: Any) -> Any:
        """The 'Implementation'. Each child must define this."""
        raise NotImplementedError
