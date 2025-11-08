from abc import ABC, abstractmethod
import logging
from pydantic import BaseModel
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


class ClassificationParameters(BaseModel):
    product: ProductItem
    user_input: Dict[str, Any]


class ClassificationWorkflow(ABC):
    """Abstract base class for independent classification workflows."""
    _max_tokens = 1

    def __init__(
            self,
            name: str,
        ):
        """Abstract base class for defining a classification workflow.
        
        Args:
            name: Name of the classification workflow.
        """
        self.name = name

    @abstractmethod
    async def _run(
            self,
            parameters: ClassificationParameters,
        ) -> ClassificationResult:
        """Runs the classification."""
        pass

    async def run(
            self,
            parameters: ClassificationParameters,
        ) -> ClassificationResult:
        """Runs the classification and writes it to the product item."""
        url = parameters.product.url
        logger.info(f'Classifying product with url={url} (workflow={self.name}).')
        try:
            classification = await self._run(parameters=parameters)
        except Exception as e:
            msg = f'Error while running classification workflow "{self.name}": {e}'
            logger.error(msg=msg)
            raise type(e)(msg) from e
        logger.info(
            f'Classification for url="{url}" (workflow={self.name}): {classification.result} and total tokens used: {classification.input_tokens + classification.output_tokens}'
        )
        return classification


class OpenAIClassification(ClassificationWorkflow):
    """Classification by OpenAI model."""

    def __init__(
            self,
            name: str,
            http_client: httpx.AsyncClient,
            api_key: str,
            model: str,
        ):
        """Open AI Chat node.

        Args:
            name: Name of the node (unique identifier)
            http_client: An httpx.AsyncClient to use for the async requests.
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
            system_prompt: System prompt for the AI model.
        """
        super().__init__(name=name)
        self._client = AsyncOpenAI(http_client=http_client, api_key=api_key)
        self._model = model

    def _log_before(self, url: str, retry_state: RetryCallState) -> None:
        """Context aware logging before the request is made."""
        if retry_state:
            logger.debug(
                f"Classifying product with url={url} with workflow={self.name} (Attempt {retry_state.attempt_number})."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before.")

    def _log_before_sleep(
        self, url: str, retry_state: RetryCallState
    ) -> None:
        """Context aware logging before sleeping after a failed request."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f"Attempt {retry_state.attempt_number} of classifying product with url={url} with workflow={self.name} "
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
            raise type(e)(f"Failed to convert OpenAI response '{content}' to integer: {e}") from e

        return ClassificationResult(
            result=content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

class OpenAIChat(OpenAIClassification):
    """Open AI classification from given product_item fields.
    
    Note:
        The system prompt sets the classes to be produced. They must be contained in allowed classes.
        The fields declared in product_item_fields are concatenated for creating a user prompt from
        which the classification should happen.
    """
    
    _user_prompt_template = "Product Details:\n{product_details}\n\nRelevance:"
    _product_details_template = "{field_name}:\n{field_value}"

    def __init__(
            self,
            name: str,
            http_client: httpx.AsyncClient,
            api_key: str,
            model: str,
            product_item_fields: List[str],
            system_prompt: str,
            allowed_classes: List[int],
        ):
        """Open AI Chat node.

        Args:
            name: Name of the node (unique identifier)
            http_client: An httpx.AsyncClient to use for the async requests.
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
            product_item_fields: Product item fields used to construct the user prompt.
            system_prompt: System prompt for the AI model.
            allowed_classes: Allowed classes for model output.
        """
        super().__init__(
            name=name,
            http_client=http_client,
            api_key=api_key,
            model=model,
        )
        self._product_item_fields = product_item_fields
        self._system_prompt = system_prompt
        self._allowed_classes = allowed_classes

    def _get_product_details(self, product: ProductItem) -> str:
        """Extracts product details based on the configuration.

        Args:
            product: The product item to extract details from.
        """
        details = []
        for name in self._product_item_fields:
            if (value := getattr(product, name, None)):
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
            parameters: ClassificationParameters,
        ) -> ClassificationResult:
        """Calls the OpenAI API with the user prompt from the product."""

        # Form the product details from the ProductItem
        product_details = self._get_product_details(product=parameters.product)
        if not product_details:
            raise KeyError(f"Missing product_details for product_item_fields={self._product_item_fields}.")

        # Create user prompt
        user_prompt = self._user_prompt_template.format(
            product_details=product_details,
        )

        # Call the OpenAI API
        url = parameters.product.url
        try:
            logger.debug(
                f"Classifying product with url={url} with workflow={self.name}."
            )
            # Perform the request and retry if necessary. There is some context aware logging
            #  - `before`: before the request is made (or before retrying)
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
                    classification = await self._call_openai_api(
                        system_prompt=self._system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=self._max_tokens,
                    )

            # Enforce that the classification is in the allowed classes
            if classification.result not in self._allowed_classes:
                logger.warning(f"Classification '{classification.result}' not in allowed classes {self._allowed_classes}")
                classification.result = PROCESSOR_DEFAULT_IF_MISSING

        except Exception as e:
            raise type(e)(f'Error classifying product at url="{url}" with workflow="{self.name}": {e}') from e

        logger.info(
            f'Classification for url="{url}" with workflow="{self.name}": {classification.result} (total tokens used={classification.input_tokens + classification.output_tokens})'
        )
        return classification


# TODO continue from here

# Typos

#   1. Line 27: "independant" → "independent"
#   2. Line 63: "worklfow" → "workflow"
#   3. Line 137: "alre" → "already" (in comment "we alre return")
#   4. Line 149: "The must be" → "They must be" (grammar issue)
#   5. Line 176: "ouptut" → "output"
#   6. Line 256: "occured" → "occurred"
#   7. Line 279: "tokensif" → "tokens if"

#   Critical Bugs

#   1. Line 220: Wrong template used (CRITICAL BUG)
#   user_prompt = self._product_details_template.format(
#       product_details=product_details,
#   )
#   1. Should be self._user_prompt_template.format(...) - this will crash because _product_details_template expects field_name and field_value, not product_details
#   2. Line 252: References undefined attribute self._workflows (never initialized in Processor.__init__)
#   3. Line 254: Missing required parameters argument
#   clsn = await wflw.run()
#   3. Should be clsn = await wflw.run(parameters=ClassificationParameters(product=product, user_input=user_input))
#   4. Line 289: No return statement but method signature declares -> ProductItem
#   5. Lines 281-287: Unused attributes in Processor.__init__:
#     - self._client (created but never used)
#     - self._model (stored but never used)
#     - self._error_response (created but never used)

#   Design Issues

#   1. Incomplete Processor.run method: Builds a classifications list but doesn't use it or return anything
#   2. Unused user_input parameter: Declared in Processor.run but never used
#   3. Missing _workflows initialization: Processor.__init__ should initialize self._workflows: List[ClassificationWorkflow] = [] or accept it as a parameter
#   4. Inconsistent constructor parameters: Processor accepts http_client, api_key, model but doesn't use them (since it doesn't create workflows itself)
#   5. Docstring inaccuracy (line 85): Mentions system_prompt parameter that doesn't exist in OpenAIClassification.__init__

#   Refactoring Suggestions

#   1. Fix Processor design: Either:
#     - Accept workflows as a constructor parameter, OR
#     - Remove unused http_client, api_key, model, _error_response parameters/attributes
#   2. Complete Processor.run: The method should either:
#     - Return a modified ProductItem with classifications attached
#     - Return the classifications list
#     - Store classifications somewhere on the product
#   3. Make _max_tokens configurable: Currently hardcoded to 1 (line 28)
#   4. Add validation: Validate that allowed_classes is non-empty and product_item_fields are valid
#   5. Type hints: Add return type hint to _get_product_details (line 188)
#   6. Consider removing duplicate logging: Lines 223-225 duplicate logging already done in ClassificationWorkflow.run (line 62-64)


# class Processor:
#     """Processes product items for a set of classification workflows."""

#     def __init__(
#         self,
#         http_client: httpx.AsyncClient,
#         api_key: str,
#         model: str,
#         default_if_missing: int = PROCESSOR_DEFAULT_IF_MISSING,
#         empty_token_count: int = PROCESSOR_EMPTY_TOKEN_COUNT,
#     ):
#         """Initializes the Processor.

#         Args:
#             http_client: An httpx.AsyncClient to use for the async requests.
#             api_key: The OpenAI API key.
#             model: The OpenAI model to use.
#             default_if_missing: The default classification to return if error occurs.
#             empty_token_count: The default value to return as tokensif the classification is empty.
#         """
#         self._client = AsyncOpenAI(http_client=http_client, api_key=api_key)
#         self._model = model
#         self._error_response = ClassificationResult(
#             result=default_if_missing,
#             input_tokens=empty_token_count,
#             output_tokens=empty_token_count,
#         )

#     async def run(self, product: ProductItem, user_input: Dict[str, Any]) -> ProductItem:
#         """Run the processing step for multiple classification workflows.
        
#         Args:
#             product: The product item to run the classification workflows for.
#         """
#         parameters = ClassificationParameters(product=product, user_input=user_input)
#         classifications = []
#         for wflw in self._workflows:
#             try:
#                 clsn = await wflw.run(parameters=parameters)
#             except Exception as e:
#                 logger.warning(f"Exception occurred when running classification workflow={wflw.name}: {e}")
#                 clsn = ClassificationResult(
#                     result=PROCESSOR_DEFAULT_IF_MISSING,
#                     input_tokens=0,
#                     output_tokens=0
#                 )
#             classifications.append(clsn)
