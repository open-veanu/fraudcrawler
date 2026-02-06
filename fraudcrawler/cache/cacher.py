import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable
from urllib.parse import urlparse

from aiocache import Cache
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
    def __init__(
        self,
        redis_use_cache: bool = REDIS_USE_CACHE,
        redis_url: str = REDIS_URL,
        redis_ttl: int = REDIS_TTL,
        redis_namespace: str = REDIS_NAMESPACE,
        redis_lease: int = REDIS_LEASE,
    ) -> None:
        self._use_cache = redis_use_cache
        self._redis_url = redis_url
        self._redis_ttl = redis_ttl
        self._redis_namespace = redis_namespace
        self._redis_lease = redis_lease
        self._serializer = _PydanticJsonSerializer()

        if not self._use_cache:
            self._wrapped = self.apply
            self._cache = None
            logger.info("Not using cache for %s", self.__class__.__name__)
            return

        self._cache_kwargs = self._redis_kwargs()

        from aiocache.backends.redis import RedisCache
        from typing import cast
        if Cache.REDIS is None:
            raise ValueError
        self._cache = cast(RedisCache, Cache(
            cache_class=Cache.REDIS,
            serializer=self._serializer,
            namespace=self._redis_namespace,
            **self._cache_kwargs,
        ))

        self._wrapped = self._cached_apply
        self._assert_redis_available()
        logger.info("Using cache for %s", self.__class__.__name__)

    def _assert_redis_available(self) -> None:
        """
        Fail-fast check.

        - If no event loop is running (typical sync construction), run the async check
          and raise immediately.
        - If an event loop is already running, schedule the check and propagate errors
          via the task callback (so no "Task exception was never retrieved").
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._redis_is_available_or_raise())
        else:
            task = loop.create_task(self._redis_is_available_or_raise())
            task.add_done_callback(self._raise_task_exception)

    @staticmethod
    def _raise_task_exception(task: "asyncio.Task[None]") -> None:
        exc = task.exception()
        if exc:
            raise exc

    async def _redis_is_available_or_raise(self) -> None:
        try:
            ok = await self.redis_is_available()
        except Exception as e:
            raise RuntimeError(
                f"Redis availability check failed for {self._redis_ttl}"
            ) from e
        if ok:
            logger.info(f"Redis is available at {self._redis_url}")
        else:
            raise RuntimeError(f"Redis is not available at {self._redis_url}")

    async def redis_is_available(self) -> bool:
        """
        Works with aiocache backends: does a small SET/GET/DEL roundtrip.
        """
        if self._cache is None:
            raise RuntimeError("Redis cache not initialized")

        key = f"__healthcheck__:{self.__class__.__name__}:{uuid.uuid4().hex}"
        value = "1"
        try:
            await self._cache.set(key=key, value=value, ttl=2)
            got = await self._cache.get(key=key)
            await self._cache.delete(key=key)
            return got == value
        except Exception:
            return False

    def _redis_kwargs(self) -> dict[str, Any]:
        u = urlparse(self._redis_url)
        if u.scheme not in {"redis", "rediss"}:
            raise ValueError("redis_url must start with redis:// or rediss://")
        return {
            "endpoint": u.hostname or "localhost",
            "port": u.port or 6379,
            "password": u.password,
            "db": int((u.path or "/0").lstrip("/") or "0"),
        }

    @staticmethod
    def _stable_key(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))

    @staticmethod
    def _serialize_for_key(obj: Any) -> Any:
        """Recursively serialize args/kwargs for cache keys. Uses model_dump() for
        Pydantic models; recurses into list and dict; leaves the rest unchanged.
        """
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, list):
            return [RedisCacher._serialize_for_key(x) for x in obj]
        if isinstance(obj, dict):
            return {k: RedisCacher._serialize_for_key(v) for k, v in obj.items()}
        return obj

    def _key_builder(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        _args = tuple(self._serialize_for_key(a) for a in args)
        _kwargs = {k: self._serialize_for_key(v) for k, v in kwargs.items()}
        return self._stable_key(
            {
                "cls": self.__class__.__name__,
                "fn": func.__name__,
                "args": _args,
                "kwargs": _kwargs,
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
        if self._cache is None:
            raise RuntimeError("Redis cache not initialized")
        key = self._key_builder(self.apply, *args, **kwargs)
        val = await self._cache.get(key=key)
        ctx = self._cache_log_context(args=args, kwargs=kwargs)
        if len(ctx) > 72:
            ctx = ctx[:69] + "..."
        suffix = f" | {ctx}" if ctx else ""
        if val is not None:
            logger.info("Cache hit [%s]%s", self.__class__.__name__, suffix)
            return val
        logger.info("Cache miss [%s]%s", self.__class__.__name__, suffix)
        result = await self.apply(*args, **kwargs)
        await self._cache.set(key=key, value=result, ttl=self._redis_ttl)
        return result

    async def capply(self, *args: Any, **kwargs: Any) -> Any:
        return await self._wrapped(*args, **kwargs)

    async def invalidate(self, *args: Any, **kwargs: Any) -> None:
        if self._use_cache:
            if self._cache is None:
                raise RuntimeError("Redis cache not initialized")
            await self._cache.delete(key=self._key_builder(self.apply, *args, **kwargs))

    async def clear_namespace(self) -> None:
        if self._use_cache:
            if self._cache is None:
                raise RuntimeError("Redis cache not initialized")
            await self._cache.clear()

    @abstractmethod
    async def apply(self, *args: Any, **kwargs: Any) -> Any:
        """The 'Implementation'. Each child must define this."""
        raise NotImplementedError
