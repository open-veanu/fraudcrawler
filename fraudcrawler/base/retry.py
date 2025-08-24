from httpx import HTTPStatusError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from fraudcrawler.settings import (
    RETRY_STOP_AFTER_ATTEMPT,
    RETRY_INITIAL_DELAY,
    RETRY_MAX_DELAY,
    RETRY_EXP_BASE,
    RETRY_JITTER,
    RETRY_SKIP_IF_CODE,
)


def _is_retryable_exception(err: BaseException) -> bool:
    if (
        isinstance(err, HTTPStatusError)
        and err.response.status_code in RETRY_SKIP_IF_CODE
    ):
        return False
    return True


def get_async_retry() -> AsyncRetrying:
    """returns the retry configuration for async operations."""
    return AsyncRetrying(
        retry=retry_if_exception(_is_retryable_exception),
        stop=stop_after_attempt(RETRY_STOP_AFTER_ATTEMPT),
        wait=wait_exponential_jitter(
            initial=RETRY_INITIAL_DELAY,
            max=RETRY_MAX_DELAY,
            exp_base=RETRY_EXP_BASE,
            jitter=RETRY_JITTER,
        ),
        reraise=True,
    )
