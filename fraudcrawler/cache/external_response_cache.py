import asyncio
import hashlib
import json
import logging
import os
from functools import wraps
from typing import Awaitable, Callable, Mapping, Optional, ParamSpec, TypeVar, cast

from aiocache import Cache
from aiocache.lock import RedLock
from aiocache.serializers import PickleSerializer

logger = logging.getLogger(__name__)

_USE_CACHE = os.getenv("REDIS_USE_CACHE", "true") == "true"
DEFAULT_TTL = int(os.getenv("REDIS_CACHE_TTL", "86400"))
DEFAULT_LOCK_LEASE = 60
WAIT_TIMEOUT = 30.0
POLL_INTERVAL = 0.2

_cache: Optional[Cache] = None

P = ParamSpec("P")
R = TypeVar("R")


def _get_cache() -> Cache:
    global _cache
    if _cache is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _cache = Cache.from_url(redis_url)
        _cache.namespace = "resp_cache"
        _cache.serializer = PickleSerializer()
        logger.info(
            f"Initialized Redis cache: {redis_url} (namespace={_cache.namespace})"
        )
    return _cache


def build_cache_key(signature_payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        signature_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_incompatible_cache_entry_error(e: Exception) -> bool:
    msg = str(e)
    name = type(e).__name__
    return ("UnpicklingError" in name) or ("invalid load key" in msg)


async def _cache_get_or_purge(cache: Cache, key: str) -> object | None:
    """
    Returns cached value if present. If entry exists but can't be deserialized due to serializer mismatch,
    delete and treat as miss.
    """
    try:
        return await cache.get(key)
    except Exception as e:
        if _is_incompatible_cache_entry_error(e):
            try:
                await cache.delete(key)
            except Exception as exc:
                logger.debug(
                    "Failed to purge incompatible cache entry key=%r (ignored): %s",
                    key,
                    exc,
                    exc_info=True,
                )
            return None
        raise


async def _wait_for_cache(
    cache: Cache, key: str, wait_timeout: float, poll_interval: float
) -> object | None:
    deadline = asyncio.get_event_loop().time() + wait_timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(poll_interval)
        try:
            v = await _cache_get_or_purge(cache, key)
            if v is not None:
                return v
        except Exception as exc:
            logger.debug(
                "Transient cache issue during wait for key=%r (ignored): %s",
                key,
                exc,
                exc_info=True,
            )
    return None


def _log_context(signature_payload: Mapping[str, object]) -> str:
    endpoint = signature_payload.get("endpoint", "?")
    provider = signature_payload.get("provider")

    # serpapi has no url parameter, so we use the search query
    if provider == "serpapi" and (q := signature_payload.get("q")):
        url = f"q={q}"
    else:
        url = (
            signature_payload.get("url") or signature_payload.get("request_url") or "?"
        )

    if provider:
        return f"provider={provider} endpoint={endpoint} url={url}"
    return f"endpoint={endpoint} url={url}"


def cached_external_call(
    *,
    key_builder: Optional[Callable[P, Mapping[str, object]]] = None,
    ttl: int = DEFAULT_TTL,
    wait_timeout: float = WAIT_TIMEOUT,
    poll_interval: float = POLL_INTERVAL,
    lock_lease: int = DEFAULT_LOCK_LEASE,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Bypass cache if disabled
            if not _USE_CACHE:
                return await func(*args, **kwargs)

            async def call_uncached() -> R:
                return await func(*args, **kwargs)

            # 1) signature payload
            if key_builder is not None:
                try:
                    signature_payload = key_builder(*args, **kwargs)
                except Exception:
                    logger.exception(
                        f"key_builder failed for {func.__name__}; bypassing cache"
                    )
                    return await call_uncached()
            else:
                sp = kwargs.get("signature_payload")
                if not isinstance(sp, Mapping):
                    logger.warning(
                        f"No/invalid signature_payload for {func.__name__}; bypassing cache"
                    )
                    return await call_uncached()
                signature_payload = cast(Mapping[str, object], sp)

            # 2) cache + key
            try:
                cache = _get_cache()
                cache_key = build_cache_key(signature_payload)
            except Exception:
                logger.exception(
                    f"Cache init/key build failed for {func.__name__}; bypassing cache"
                )
                return await call_uncached()

            ctx = _log_context(signature_payload)

            # 3) fast path
            try:
                cached_value = await _cache_get_or_purge(cache, cache_key)
            except Exception:
                logger.exception(
                    f"Cache get failed for {func.__name__}; bypassing cache ({ctx})"
                )
                return await call_uncached()

            if cached_value is not None:
                logger.info(f"Using cache for {ctx}")
                return cast(R, cached_value)

            logger.info(f"Cache miss for {ctx}")

            # 4) lock + compute
            lock_key = f"{cache_key}:lock"
            try:
                async with RedLock(cache, lock_key, lease=lock_lease):
                    # re-check after lock
                    cached_value = await _cache_get_or_purge(cache, cache_key)
                    if cached_value is not None:
                        logger.info(f"Using cache (LOCK-RACE-HIT) for {ctx}")
                        return cast(R, cached_value)

                    result = await call_uncached()  # do not cache failures

                    try:
                        await cache.set(cache_key, result, ttl=ttl)
                        logger.info(f"Computed and cached result for {ctx}")
                    except Exception:
                        logger.exception(
                            f"Cache set failed for {func.__name__} (continuing) ({ctx})"
                        )

                    return result
            except Exception:
                logger.debug(
                    f"Lock path failed for {func.__name__}; waiting up to {wait_timeout:.1f}s ({ctx})"
                )

            # 5) wait/poll
            v = await _wait_for_cache(cache, cache_key, wait_timeout, poll_interval)
            if v is not None:
                logger.info(f"Using cache (WAIT-HIT) for {ctx}")
                return cast(R, v)

            # 6) fallback compute
            logger.info(
                f"Cache unavailable (timeout); computing without cache for {ctx}"
            )
            result = await call_uncached()
            try:
                await cache.set(cache_key, result, ttl=ttl)
                logger.info(f"Computed and cached result (post-timeout) for {ctx}")
            except Exception as exc:
                logger.debug(
                    "Post-timeout cache set failed for %s (ignored): %s",
                    ctx,
                    exc,
                    exc_info=True,
                )
            return result

        return wrapper

    return decorator
