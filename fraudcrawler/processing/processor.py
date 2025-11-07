from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List

import httpx
from openai import AsyncOpenAI
from tenacity import RetryCallState

from fraudcrawler.base.base import ProductItem, Prompt, ClassificationResult
from fraudcrawler.base.retry import get_async_retry
from fraudcrawler.settings import (
    PROCESSOR_DEFAULT_IF_MISSING,
    PROCESSOR_EMPTY_TOKEN_COUNT,
)


logger = logging.getLogger(__name__)


class ClassificationWorkflow(ABC):
    """Abstract base class for independant classification workflows."""


    def __init__(
            self,
            name: str,
        ):
        """Abstract base class for defining a classification workflow.
        
        Args:
            name: Name of the classification workflow.
        """
        self._name = name

    @abstractmethod
    async def _run(
            self,
            product: ProductItem,
            user_input: Dict[str, Any] | None = None,
        ) -> ClassificationResult:
        """Runs the classification.

        Args:
            product: Product item to process.
            user_input: Additional user input data.
        """
        pass

    async def run(
            self,
            product: ProductItem,
            user_input: Dict[str, Any] | None = None,
        ) -> ClassificationResult:
        """Runs the classification and writes it to the product item.
        
        Args:
            product: Product item to process.
            user_input: Additional user input data.
        """
        logger.info(f'Start classification workflow {self._name}')
        try:
            classification = await self._run(product=product, user_input=user_input)
        except Exception as e:
            msg = f'Error while running classification workflow "{self._name}": {e}'
            logger.error(msg=msg)
            raise type(e)(msg) from e
        logger.info(f'Run of node {self._name} finished successfully.')
        return classification


class OpenAIChat(ClassificationWorkflow):
    """Open AI Chat model."""

    _user_prompt_template = "Product Details:\n{product_details}\n\nRelevance:"
    _product_details_template = "{field_name}:\n{field_value}"
    _default_allowed_classes = [0, 1]
    _max_tokens = 1

    def __init__(
            self,
            name: str,
            http_client: httpx.AsyncClient,
            api_key: str,
            model: str,
            system_prompt: str,
            product_item_fields: List[str],
            allowed_classes: List[int] | None = None,
        ):
        """Open AI Chat node.

        Args:
            name: Name of the node (unique identifier)
            http_client: An httpx.AsyncClient to use for the async requests.
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
            system_prompt: System prompt for the AI model.
            product_item_fields: Product item fields used to construct the user prompt.
            allowed_classes: Allowed classification values.
        """
        super().__init__(name=name)
        self._client = AsyncOpenAI(http_client=http_client, api_key=api_key)
        self._model = model
        self._system_prompt = system_prompt
        self._product_item_fields = product_item_fields
        self._allowed_classes = allowed_classes if allowed_classes is not None else self._default_allowed_classes

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

    def _get_product_details(self, product: ProductItem) -> str:
        """Extracts product details based on the configuration.

        Args:
            product: The product item to extract details from.
        """
        details = []
        for name in self._product_item_fields:
            if value := getattr(product, name, None):
                details.append(
                    self._product_details_template.format(
                        field_name=name, field_value=value
                    )
                )
            else:
                logger.warning(
                    f'Field "{name}" is missing in ProductItem with url="{product.url}"'
                )
        return "\n\n".join(details)

    async def _run(
            self,
            product: ProductItem,
            user_input: Dict[str, Any] | None = None,
        ) -> ClassificationResult:
        """Calls the OpenAI API with the user prompt from the product."""

        # Form the product details from the ProductItem
        product_details = self._get_product_details(product=product)
        if not product_details:
            logger.warning("Missing required product_details for classification.")
            return self._error_response

        # Prepare the user prompt
        user_prompt = PROCESSOR_USER_PROMPT_TEMPLATE.format(
            product_details=product_details,
        )


        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
        )
        if not response or not (content := response.choices[0].message.content):
            raise ValueError(
                f'Error calling OpenAI API or empty response="{response}".'
            )

        # Convert the content to an integer
        try:
            content = int(content.strip())
        except Exception as e:
            raise type(e)(f"Failed to convert OpenAI response '{content}' to integer: {e}") from e

        # For tracking consumption we alre return the tokens used
        classification = ClassificationResult(
            result=content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        return classification

class Processor:
    """Processes product items for a set of classification workflows."""

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

    async def run(self, product: ProductItem) -> ProductItem:
        """Run the processing step for multiple classification workflows.
        
        Args:
            product: The product item to run the classification workflows for.
        """
        for workflow in self._classification_workflows:
            try:
                workflow.run()
            except:


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
