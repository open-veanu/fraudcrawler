from aiohttp.web_exceptions import HTTPException
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from fraudcrawler.settings import (
    RETRY_STOP_AFTER_ATTEMPT,
    RETRY_STOP_AFTER_ATTEMPT_SYNC,
    RETRY_INITIAL_DELAY,
    RETRY_MAX_DELAY,
    RETRY_EXP_BASE,
    RETRY_JITTER,
    RETRY_SKIP_IF_CODE,
)


def _is_retryable_exception(err: BaseException) -> bool:
    if isinstance(err, HTTPException) and err.status_code in RETRY_SKIP_IF_CODE:
        return False
    return True


_RETRY_KWARGS = {
    "retry": retry_if_exception(_is_retryable_exception),
    "stop": stop_after_attempt(RETRY_STOP_AFTER_ATTEMPT),
    "wait": wait_exponential_jitter(
        initial=RETRY_INITIAL_DELAY,
        max=RETRY_MAX_DELAY,
        exp_base=RETRY_EXP_BASE,
        jitter=RETRY_JITTER,
    ),
    "reraise": True,
}

def get_async_retry() -> AsyncRetrying:
    """returns the retry configuration for async operations."""
    return AsyncRetrying(**_RETRY_KWARGS)


def get_sync_retry() -> Retrying:
    """returns the retry configuration for synchronous operations."""
    return Retrying(**_RETRY_KWARGS)
