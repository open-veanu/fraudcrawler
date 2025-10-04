from base64 import b64decode
import logging
from typing import List

from bs4 import BeautifulSoup
import httpx
from tenacity import RetryCallState

from fraudcrawler.settings import ZYTE_DEFALUT_PROBABILITY_THRESHOLD
from fraudcrawler.base.base import DomainUtils, ProductItem
from fraudcrawler.base.retry import get_async_retry

logger = logging.getLogger(__name__)


class ZyteAPI(DomainUtils):
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
        http_client: httpx.AsyncClient,
        api_key: str,
    ):
        """Initializes the ZyteApiClient with the given API key and retry configurations.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            api_key: The API key for Zyte API.
        """
        self._http_client = http_client
        self._api_key = api_key

    def _log_before(self, url: str, retry_state: RetryCallState | None) -> None:
        """Context aware logging before the request is made."""
        if retry_state:
            logger.debug(
                f"Zyte fetching product details for URL {url} (Attempt {retry_state.attempt_number})."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before.")

    def _log_before_sleep(self, url: str, retry_state: RetryCallState | None) -> None:
        """Context aware logging before sleeping after a failed request."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f'Attempt {retry_state.attempt_number} of Zyte fetching product details for URL "{url}" '
                f"Retrying in {retry_state.upcoming_sleep:.0f} seconds."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before_sleep.")

    @staticmethod
    def _extract_product_name(details: dict) -> str | None:
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
    def _extract_url_resolved(details: dict) -> str | None:
        """Extracts the resolved URL from the product data - this is automatically resolved by Zyte.

        The input argument is a dictionary of the following structure:
            {
                "product": {
                    "url": str,
                }
            }
        """
        return details.get("product", {}).get("url")

    @staticmethod
    def _extract_product_price(details: dict) -> str | None:
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
    def _extract_product_description(details: dict) -> str | None:
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
    def _extract_image_urls(details: dict) -> List[str]:
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
    def _extract_probability(details: dict) -> float:
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
        return float(
            details.get("product", {}).get("metadata", {}).get("probability", 0.0)
        )

    @staticmethod
    def _extract_html(details: dict) -> str | None:
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
        return None

    def enrich_context(self, product: ProductItem, details: dict) -> ProductItem:
        product.product_name = self._extract_product_name(details=details)

        url_resolved = self._extract_url_resolved(details=details)
        if url_resolved:
            product.url_resolved = url_resolved

        # If the resolved URL is different from the original URL, we also need to update the domain as
        # otherwise the unresolved domain will be shown.
        # For example for an unresolved domain "toppreise.ch" but resolved "digitec.ch
        if url_resolved and url_resolved != product.url:
            logger.debug(f"URL resolved for {product.url} is {url_resolved}")
            product.domain = self._get_domain(url=url_resolved)

        product.product_price = self._extract_product_price(details=details)
        product.product_description = self._extract_product_description(details=details)
        product.product_images = self._extract_image_urls(details=details)
        product.probability = self._extract_probability(details=details)
        product.html = self._extract_html(details=details)
        if product.html:
            soup = BeautifulSoup(product.html, "html.parser")
            product.html_clean = soup.get_text(separator=" ", strip=True)

        return product

    @staticmethod
    def keep_product(
        details: dict,
        threshold: float = ZYTE_DEFALUT_PROBABILITY_THRESHOLD,
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

    async def unblock_url_content(self, url: str) -> bytes:
        """Unblock the content of an URL using Zyte proxy mode.

        Args:
            url: The URL to fetch using Zyte proxy mode.
        """
        logger.debug(f'Unblock URL content using Zyte proxy for url="{url}"')
        details = await self.details(url)

        if not details or "httpResponseBody" not in details:
            raise httpx.HTTPError("No httpResponseBody in Zyte response")

        return b64decode(details["httpResponseBody"])

    async def details(self, url: str) -> dict:
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

        # Perform the request and retry if necessary. There is some context aware logging:
        #  - `before`: before the request is made (and before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            url=url, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            url=url, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                response = await self._http_client.post(
                    url=self._endpoint,
                    json={"url": url, **self._config},
                    auth=(self._api_key, ""),  # API key as username, empty password
                )
                response.raise_for_status()

        details = response.json()
        return details
