import logging

import httpx
from openai import AsyncOpenAI
from tenacity import RetryCallState

from fraudcrawler.base.base import ProductItem, Prompt, ClassificationResult
from fraudcrawler.base.retry import get_async_retry
from fraudcrawler.settings import (
    PROCESSOR_PRODUCT_DETAILS_TEMPLATE,
    PROCESSOR_USER_PROMPT_TEMPLATE,
    PROCESSOR_DEFAULT_IF_MISSING,
    PROCESSOR_EMPTY_TOKEN_COUNT,
)


logger = logging.getLogger(__name__)


class Processor:
    """Processes product data for classification based on a prompt configuration."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        api_key: str,
        model: str,
        default_if_missing: int = PROCESSOR_DEFAULT_IF_MISSING,
        empty_token_count: int = PROCESSOR_EMPTY_TOKEN_COUNT,
    ):
        """Initializes the Processor.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
            default_if_missing: The default classification to return if error occurs.
            empty_token_count: The default value to return as tokensif the classification is empty.
        """
        self._client = AsyncOpenAI(http_client=http_client, api_key=api_key)
        self._model = model
        self._error_response = ClassificationResult(
            result=default_if_missing,
            input_tokens=empty_token_count,
            output_tokens=empty_token_count,
        )

    @staticmethod
    def _get_product_details(product: ProductItem, prompt: Prompt) -> str:
        """Extracts product details based on the prompt configuration.

        Args:
            product: The product item to extract details from.
            prompt: The prompt configuration containing field names.
        """
        details = []
        for field in prompt.product_item_fields:
            if value := getattr(product, field, None):
                details.append(
                    PROCESSOR_PRODUCT_DETAILS_TEMPLATE.format(
                        field_name=field, field_value=value
                    )
                )
            else:
                logger.warning(
                    f'Field "{field}" is missing in ProductItem with url="{product.url}"'
                )
        return "\n\n".join(details)

    @staticmethod
    def _log_before(url: str, prompt: Prompt, retry_state: RetryCallState) -> None:
        """Context aware logging before the request is made."""
        if retry_state:
            logger.debug(
                f"Classifying product with url={url} using prompt={prompt.name} (Attempt {retry_state.attempt_number})."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before.")

    @staticmethod
    def _log_before_sleep(
        url: str, prompt: Prompt, retry_state: RetryCallState
    ) -> None:
        """Context aware logging before sleeping after a failed request."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f"Attempt {retry_state.attempt_number} of classifying product with url={url} using prompt={prompt.name} "
                f"failed with error: {retry_state.outcome.exception()}. "
                f"Retrying in {retry_state.upcoming_sleep:.0f} seconds."
            )

    async def _call_openai_api(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> ClassificationResult:
        """Calls the OpenAI API with the given user prompt."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        if not response or not (content := response.choices[0].message.content):
            raise ValueError(
                f'Error calling OpenAI API or empty response="{response}".'
            )

        # Convert the content to an integer
        try:
            content = int(content.strip())
        except Exception as e:
            msg = f"Failed to convert OpenAI response '{content}' to integer: {e}"
            logger.error(msg)
            raise ValueError(msg)

        # For tracking consumption we alre return the tokens used
        classification = ClassificationResult(
            result=content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        return classification

    async def classify(
        self,
        product: ProductItem,
        prompt: Prompt,
    ) -> ClassificationResult:
        """A generic classification method that classifies a product based on a prompt object and returns
          the classification, input tokens, and output tokens.

        Args:
            product: The product item to classify.
            prompt: The prompt to use for classification.

        Note:
            This method returns `PROCESSOR_DEFAULT_IF_MISSING` if:
                - product_details is empty
                - an error occurs during the API call
                - if the response isn't in allowed_classes.
        """
        url = product.url

        # Form the product details from the ProductItem
        product_details = self._get_product_details(product=product, prompt=prompt)
        if not product_details:
            logger.warning("Missing required product_details for classification.")
            return self._error_response

        # Prepare the user prompt
        user_prompt = PROCESSOR_USER_PROMPT_TEMPLATE.format(
            product_details=product_details,
        )

        # Call the OpenAI API
        try:
            logger.debug(
                f"Classifying product with url={url}, using prompt={prompt.name}."
            )
            # Perform the request and retry if necessary. There is some context aware logging
            #  - `before`: before the request is made (or before retrying)
            #  - `before_sleep`: if the request fails before sleeping
            retry = get_async_retry()
            retry.before = lambda retry_state: self._log_before(
                url=url, prompt=prompt, retry_state=retry_state
            )
            retry.before_sleep = lambda retry_state: self._log_before_sleep(
                url=url, prompt=prompt, retry_state=retry_state
            )
            async for attempt in retry:
                with attempt:
                    classification = await self._call_openai_api(
                        system_prompt=prompt.system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=1,
                    )

            # Enforce that the classification is in the allowed classes
            if classification.result not in prompt.allowed_classes:
                logger.warning(
                    f"Classification '{classification.result}' not in allowed classes {prompt.allowed_classes}"
                )
                return self._error_response

            logger.info(
                f'Classification for url="{url}" (prompt={prompt.name}): {classification.result} and total tokens used: {classification.input_tokens + classification.output_tokens}'
            )
            return classification

        except Exception as e:
            logger.error(
                f'Error classifying product at url="{url}" with prompt "{prompt.name}": {e}'
            )
            return self._error_response
