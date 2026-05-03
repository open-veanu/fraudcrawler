import pickle  # nosec B403  # values are produced by our own dumps() and read from our own Redis
import zlib
from typing import Any, ClassVar

from aiocache.serializers import BaseSerializer

from fraudcrawler.settings import REDIS_COMPRESSION_LEVEL


class CompressedPickleSerializer(BaseSerializer):
    """Pickle + zlib serializer for aiocache Redis values."""

    # Required: aiocache's Redis backend calls value.decode(encoding) on every
    # read unless encoding is None. Without this, binary pickle/zlib bytes
    # would raise UnicodeDecodeError on every GET.
    DEFAULT_ENCODING: ClassVar[str | None] = None

    def dumps(self, value: Any) -> bytes:
        return zlib.compress(pickle.dumps(value), level=REDIS_COMPRESSION_LEVEL)

    def loads(self, value: bytes | None) -> Any:
        if value is None:
            return None
        return pickle.loads(zlib.decompress(value))  # noqa: S301  # nosec B301