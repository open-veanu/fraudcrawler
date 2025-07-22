import asyncio
import logging
from typing import List
from base64 import b64decode

import aiohttp

from fraudcrawler.settings import (
    MAX_RETRIES,
    RETRY_DELAY,
    ZYTE_DEFALUT_PROBABILITY_THRESHOLD,
)
from fraudcrawler.base.base import AsyncClient

logger = logging.getLogger(__name__)


class ZyteApi(AsyncClient):
    """A client to interact with the Zyte API for fetching product details."""

    _endpoint = "https://api.zyte.com/v1/extract"
    _config = {
        "javascript": False,
        "browserHtml": False,
        "screenshot": False,
        "productOptions": {"extractFrom": "httpResponseBody"},
        "httpResponseBody": True,
        "geolocation": "CH",
        "viewport": {"width": 1280, "height": 1080},
        "product": True,
        # "actions": [],
    }

    def __init__(
        self,
        api_key: str,
        max_retries: int = MAX_RETRIES,
        retry_delay: int = RETRY_DELAY,
    ):
        """Initializes the ZyteApiClient with the given API key and retry configurations.

        Args:
            api_key: The API key for Zyte API.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Delay between retries in seconds.
        """
        self._aiohttp_basic_auth = aiohttp.BasicAuth(api_key)
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def get_details(self, url: str) -> dict:
        """Fetches product details for a single URL.

        Args:
            url: The URL to fetch product details from.

        Returns:
            A dictionary containing the product details, fields include:
            (c.f. https://docs.zyte.com/zyte-api/usage/reference.html#operation/extract/response/200/product)
            {
                "url": str,
                "statusCode": str,
                "product": {
                    "name": str,
                    "price": str,
                    "mainImage": {"url": str},
                    "images": [{"url": str}],
                    "description": str,
                    "metadata": {
                        "probability": float,
                    },
                },
                "httpResponseBody": base64
            }
        """
        logger.info(f"Fetching product details by Zyte for URL {url}.")
        attempts = 0
        err = None
        while attempts < self._max_retries:
            try:
                logger.debug(
                    f"Fetch product details for URL {url} (Attempt {attempts + 1})."
                )
                product = await self.post(
                    url=self._endpoint,
                    data={"url": url, **self._config},
                    auth=self._aiohttp_basic_auth,
                )
                return product
            except Exception as e:
                logger.debug(
                    f"Exception occurred while fetching product details for URL {url} (Attempt {attempts + 1})."
                )
                err = e
            attempts += 1
            if attempts < self._max_retries:
                await asyncio.sleep(self._retry_delay)
        if err is not None:
            raise err
        return {}

    @staticmethod
    def keep_product(
        details: dict, threshold: float = ZYTE_DEFALUT_PROBABILITY_THRESHOLD
    ) -> bool:
        """Determines whether to keep the product based on the probability threshold.

        Args:
            details: A product details data dictionary.
            threshold: The probability threshold used to filter the products.
        """
        try:
            prob = float(details["product"]["metadata"]["probability"])
        except KeyError:
            logger.warning(
                f"Product with url={details.get('url')} has no probability value - product is ignored"
            )
            return False
        return prob > threshold

    @staticmethod
    def extract_product_name(details: dict) -> str | None:
        """Extracts the product name from the product data.

        The input argument is a dictionary of the following structure:
            {
                "product": {
                    "name": str,
                }
            }
        """
        return details.get("product", {}).get("name")

    @staticmethod
    def extract_product_price(details: dict) -> str | None:
        """Extracts the product price from the product data.

        The input argument is a dictionary of the following structure:
            {
                "product": {
                    "price": str,
                }
            }
        """
        return details.get("product", {}).get("price")

    @staticmethod
    def extract_product_description(details: dict) -> str | None:
        """Extracts the product description from the product data.

        The input argument is a dictionary of the following structure:
            {
                "product": {
                    "description": str,
                }
            }
        """
        return details.get("product", {}).get("description")

    @staticmethod
    def extract_image_urls(details: dict) -> List[str]:
        """Extracts the images from the product data.

        The input argument is a dictionary of the following structure:
            {
                "product": {
                    "mainImage": {"url": str},
                    "images": [{"url": str}],
                }
            }
        """
        images = []
        product = details.get("product")
        if product:
            # Extract main image URL
            if (main_img := product.get("mainImage")) and (url := main_img.get("url")):
                images.append(url)
            # Extract additional image URLs
            if urls := product.get("images"):
                images.extend([img["url"] for img in urls if img.get("url")])
        return images

    @staticmethod
    def extract_probability(details: dict) -> float:
        """Extracts the probability from the product data.

        The input argument is a dictionary of the following structure:
            {
                "product": {
                    "metadata": {
                        "probability": float,
                    }
                }
            }
        """
        return float(details.get("product", {}).get("metadata", {}).get("probability"))

    @staticmethod
    def extract_html(details: dict) -> str | None:
        """Extracts the HTML from the Zyte API response.

        The input argument is a dictionary of the following structure:
            {
                "httpResponseBody": base64
            }
        """

        # Get the Base64-encoded content
        encoded = details.get("httpResponseBody")

        # Decode it into bytes
        if isinstance(encoded, str):
            decoded_bytes = b64decode(encoded)

        # Convert bytes to string (assuming UTF-8 encoding)
        decoded_string = decoded_bytes.decode("utf-8")
        return decoded_string
