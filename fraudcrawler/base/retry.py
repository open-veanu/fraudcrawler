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


def get_async_retry(
    stop_after: int = RETRY_STOP_AFTER_ATTEMPT,
    initial_delay: int = RETRY_INITIAL_DELAY,
    max_delay: int = RETRY_MAX_DELAY,
    exp_base: int = RETRY_EXP_BASE,
    jitter: int = RETRY_JITTER,
) -> AsyncRetrying:
    """returns the retry configuration for async operations."""
    return AsyncRetrying(
        retry=retry_if_exception(_is_retryable_exception),
        stop=stop_after_attempt(stop_after),
        wait=wait_exponential_jitter(
            initial=initial_delay,
            max=max_delay,
            exp_base=exp_base,
            jitter=jitter,
        ),
        reraise=True,
    )
