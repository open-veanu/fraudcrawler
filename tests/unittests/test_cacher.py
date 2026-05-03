import pickle
import uuid
import zlib
from typing import cast

import pytest
import pytest_asyncio
from aiocache import Cache
from aiocache.backends.redis import RedisCache
from pydantic import BaseModel

from fraudcrawler.cache.serializers import CompressedPickleSerializer


REDIS_TEST_HOST = "localhost"
REDIS_TEST_PORT = 6379
REDIS_TEST_DB = 15
REDIS_TEST_TTL = 30


class _DummyModel(BaseModel):
    name: str
    count: int
    payload: bytes


# ---------------------------------------------------------------------------
# Pure serializer round-trip tests (no Redis)
# ---------------------------------------------------------------------------


def test_round_trip_str() -> None:
    serializer = CompressedPickleSerializer()
    value = "hello world"

    assert serializer.loads(serializer.dumps(value)) == value


def test_round_trip_dict() -> None:
    serializer = CompressedPickleSerializer()
    value = {"a": 1, "b": "two", "c": [1, 2, 3]}

    assert serializer.loads(serializer.dumps(value)) == value


def test_round_trip_nested_dict_with_bytes() -> None:
    serializer = CompressedPickleSerializer()
    value = {
        "outer": {
            "inner": {"raw": b"\x00\x01\x02binary\xff"},
            "list": [b"a", b"b"],
        },
        "top_bytes": b"top-level",
    }

    assert serializer.loads(serializer.dumps(value)) == value


def test_round_trip_pydantic_model() -> None:
    serializer = CompressedPickleSerializer()
    value = _DummyModel(name="x", count=42, payload=b"\x00\x01")

    restored = serializer.loads(serializer.dumps(value))

    assert isinstance(restored, _DummyModel)
    assert restored == value


def test_loads_none_returns_none() -> None:
    serializer = CompressedPickleSerializer()

    assert serializer.loads(None) is None


def test_compression_ratio_html_payload() -> None:
    """Repetitive HTML should compress at least 3x vs raw pickle bytes."""
    serializer = CompressedPickleSerializer()
    chunk = "<div class='x'>" + ("abc " * 50) + "</div>"
    html = chunk * 2400  # ~530 KB of repetitive HTML
    assert len(html) >= 500_000

    compressed = serializer.dumps(html)
    raw_pickle = pickle.dumps(html)

    assert len(compressed) * 3 <= len(raw_pickle), (
        f"compressed={len(compressed)} raw_pickle={len(raw_pickle)}"
    )


# ---------------------------------------------------------------------------
# Integration tests: exercise the full aiocache + Redis path with the
# compressed serializer to confirm the bytes flow end-to-end through Redis.
# Skipped automatically when Redis is not reachable on localhost.
# ---------------------------------------------------------------------------


async def _redis_available() -> bool:
    probe = cast(
        RedisCache,
        Cache(
            cache_class=Cache.REDIS,  # type: ignore[reportArgumentType]
            endpoint=REDIS_TEST_HOST,
            port=REDIS_TEST_PORT,
            db=REDIS_TEST_DB,
            namespace="fraudcrawler:test:probe",
            timeout=2,
        ),
    )
    key = f"__probe__:{uuid.uuid4().hex}"
    try:
        await probe.set(key=key, value="1", ttl=5)
        await probe.delete(key=key)
        return True
    except Exception:
        return False


@pytest_asyncio.fixture
async def redis_cache():
    if not await _redis_available():
        pytest.skip(f"Redis not reachable at {REDIS_TEST_HOST}:{REDIS_TEST_PORT}")

    namespace = f"fraudcrawler:test:cacher:{uuid.uuid4().hex}"
    cache = cast(
        RedisCache,
        Cache(
            cache_class=Cache.REDIS,  # type: ignore[reportArgumentType]
            serializer=CompressedPickleSerializer(),
            endpoint=REDIS_TEST_HOST,
            port=REDIS_TEST_PORT,
            db=REDIS_TEST_DB,
            namespace=namespace,
            timeout=5,
        ),
    )
    try:
        yield cache
    finally:
        # Drop everything we wrote under this namespace.
        await cache.clear()


@pytest.mark.asyncio
async def test_redis_round_trip_dict(redis_cache: RedisCache) -> None:
    key = f"dict:{uuid.uuid4().hex}"
    value = {"a": 1, "b": "two", "c": [1, 2, 3], "blob": b"\x00\xff"}

    await redis_cache.set(key=key, value=value, ttl=REDIS_TEST_TTL)
    restored = await redis_cache.get(key=key)

    assert restored == value


@pytest.mark.asyncio
async def test_redis_round_trip_pydantic(redis_cache: RedisCache) -> None:
    key = f"model:{uuid.uuid4().hex}"
    value = _DummyModel(name="x", count=42, payload=b"\x00\x01")

    await redis_cache.set(key=key, value=value, ttl=REDIS_TEST_TTL)
    restored = await redis_cache.get(key=key)

    assert isinstance(restored, _DummyModel)
    assert restored == value


@pytest.mark.asyncio
async def test_redis_stored_bytes_are_compressed(redis_cache: RedisCache) -> None:
    """Confirm what hits Redis is zlib-compressed, not raw pickle."""
    key = f"compressed:{uuid.uuid4().hex}"
    chunk = "<div class='x'>" + ("abc " * 50) + "</div>"
    html = chunk * 2400

    await redis_cache.set(key=key, value=html, ttl=REDIS_TEST_TTL)

    # Pull the raw bytes via aiocache's _raw escape hatch (encoding=None
    # to skip utf-8 decode), confirm zlib round-trip works, and confirm
    # the stored payload is dramatically smaller than raw pickle.
    namespaced = redis_cache.build_key(key)
    raw = await redis_cache._raw("get", namespaced, encoding=None)
    assert isinstance(raw, bytes)
    assert pickle.loads(zlib.decompress(raw)) == html  # noqa: S301
    assert len(raw) * 3 <= len(pickle.dumps(html))

    restored = await redis_cache.get(key=key)
    assert restored == html