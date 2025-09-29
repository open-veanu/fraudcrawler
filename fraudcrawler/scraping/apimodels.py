import logging
from typing import List, cast, Dict
from base64 import b64decode

import aiohttp
from tenacity import RetryCallState

from fraudcrawler.base.base import AsyncClient, DomainUtils
from fraudcrawler.base.retry import get_async_retry
from fraudcrawler.settings import VEANU_DEFAULT_PROBABILITY_THRESHOLD

logger = logging.getLogger(__name__)


class ApiModels(AsyncClient, DomainUtils):
    """A client to interact with the Custom API for fetching product details."""

    def __init__(
        self,
        apimodels_key: str,
    ):
        """Initializes the ApiModels client with the given API key and retry configurations.

        Args:
            apimodels_key: The API key for ApiModels API.
        """
        self._endpoint = "https://confidential-fraudcrawler-models-api.fly.dev/predict"
        self._aiohttp_basic_auth = aiohttp.BasicAuth(apimodels_key)
    

    def _log_before(self, product_text: str, retry_state: RetryCallState | None) -> None:
        """Context aware logging before the request is made."""
        if retry_state:
            logger.debug(
                f"ApiModels fetching product classification for product text --{product_text[0:50]}-- (Attempt {retry_state.attempt_number})."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before.")

    def _log_before_sleep(self, product_text: str, retry_state: RetryCallState | None) -> None:
        """Context aware logging before sleeping after a failed request."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f'Attempt {retry_state.attempt_number} of ApiModels fetching product details for product_text --"{product_text[0:100]}"-- '
                f"Retrying in {retry_state.upcoming_sleep:.0f} seconds."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before_sleep.")

    async def get_details_from_text(self, product_text: str) -> dict:
        """Fetches product details for a single text.

        Args:
            product_text: The text to fetch product details from.

        Returns:
            A dictionary containing the product details, fields include:
            {
                "text": str,
                "prediction": int,
                "confidence": float,
                "label": str
            }
        """
        logger.info(f"Fetching prediction from ApiModels for text: {product_text[:50]}...")

        # Perform the request and retry if necessary. There is some context aware logging:
        #  - `before`: before the request is made (and before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            product_text=product_text, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            product_text=product_text, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                # Send JSON data with authentication
                product = cast(
                    Dict,
                    await self.post(
                        url=self._endpoint,
                        data={"text": product_text},
                        headers={"Content-Type": "application/json"},
                        auth=self._aiohttp_basic_auth,
                    ),
                )
        return product

