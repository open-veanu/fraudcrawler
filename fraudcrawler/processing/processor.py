from abc import ABC, abstractmethod
from enum import Enum
import logging
from pydantic import BaseModel
from typing import Any, Dict, List

import httpx
from openai import AsyncOpenAI
from tenacity import RetryCallState

from fraudcrawler.base.base import ProductItem
from fraudcrawler.base.retry import get_async_retry
from fraudcrawler.settings import (
    PROCESSOR_DEFAULT_IF_MISSING,
    PROCESSOR_EMPTY_TOKEN_COUNT,
)


logger = logging.getLogger(__name__)


class ClfnParams(BaseModel):
    product: ProductItem
    user_input: Dict[str, Any]

class ClfnStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"

class ClfnResults(BaseModel):
    """Model for classification results."""

    result: int
    input_tokens: int
    output_tokens: int
    status: ClfnStatus


class ClfnWorkflow(ABC):
    """Abstract base class for independent classification workflows."""
    _max_tokens: int = 1

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
            parameters: ClfnParams,
        ) -> ClfnResults:
        """Runs the classification."""
        pass

    async def run(
            self,
            parameters: ClfnParams,
        ) -> ClfnResults:
        """Runs the classification and writes it to the product item."""
        url = parameters.product.url
        logger.info(f'Classifying product with url={url} (workflow={self.name}).')
        
        # Run classification (error is catched in processor.run())
        clfn = await self._run(parameters=parameters)
        
        logger.info(
            f'Classification for url="{url}" (workflow={self.name}): result={clfn.result}, status={clfn.status}, and total tokens used={clfn.input_tokens + clfn.output_tokens}'
        )
        return clfn


class OpenAIWorkflow(ClfnWorkflow):
    """Classification workflow using OpenAI API calls."""

    def __init__(
            self,
            name: str,
            http_client: httpx.AsyncClient,
            api_key: str,
            model: str,
        ):
        """Open AI Chat Workflow.

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
    ) -> ClfnResults:
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

        return ClfnResults(
            result=content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            status=ClfnStatus.SUCCESS,
        )

class OpenAIChat(OpenAIWorkflow):
    """Open AI classification workflow with single API call using specific product_item fields for setting up the context.
    
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
            parameters: ClfnParams,
        ) -> ClfnResults:
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
                    clfn = await self._call_openai_api(
                        system_prompt=self._system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=self._max_tokens,
                    )

            # Enforce that the classification is in the allowed classes
            if clfn.result not in self._allowed_classes:
                logger.warning(f"Classification '{clfn.result}' not in allowed classes {self._allowed_classes}")
                clfn.result = PROCESSOR_DEFAULT_IF_MISSING
                clfn.status = ClfnStatus.ERROR

        except Exception as e:
            raise type(e)(f'Error classifying product at url="{url}" with workflow="{self.name}": {e}') from e

        return clfn


class Processor:
    """Processes product items for a set of classification workflows."""

    def __init__(
            self,
            workflows: List[ClfnWorkflow]
        ):
        """Initializes the Processor.

        Args:
            workflows: List of classification workflows through which the processor will go.
        """
        if not self._are_unique(workflows=workflows):
            raise ValueError(f'workflow names must be unique, they are {[wf.name for wf in workflows]}')
        self._workflows = workflows
    
    @staticmethod
    def _are_unique(workflows: List[ClfnWorkflow]) -> bool:
        """Tests if the workflows have unique names."""
        return len(workflows) == len(set([wf.name for wf in workflows]))

    async def run(self, product: ProductItem, user_input: Dict[str, Any]) -> Dict[str, int]:
        """Run the processing step for multiple classification workflows.
        
        Args:
            product: The product item to run the classification workflows for.
        """
        parameters = ClfnParams(product=product, user_input=user_input)
        clfns = {}
        for wf in self._workflows:
            try:
                clfn = await wf.run(parameters=parameters)
            except Exception as e:
                logger.error(f'Error while running classification workflow="{wf.name}": {e}')
                clfn = ClfnResults(
                    result=PROCESSOR_DEFAULT_IF_MISSING,
                    input_tokens=PROCESSOR_EMPTY_TOKEN_COUNT,
                    output_tokens=PROCESSOR_EMPTY_TOKEN_COUNT,
                    status=ClfnStatus.ERROR,
                )
            clfns[wf.name] = clfn
        return clfns
        
