import logging
from pydantic import BaseModel
from typing import Dict, List, Literal, TypeAlias

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from openai.types.responses import (
    Response,
    ParsedResponse,
    ResponseInputImageParam,
    ResponseInputParam,
)
from tenacity import RetryCallState

from fraudcrawler.base.base import ProductItem
from fraudcrawler.base.retry import get_async_retry
from fraudcrawler.processing.base import (
    ClassificationResult,
    UserInputs,
    Workflow,
)

logger = logging.getLogger(__name__)

Context: TypeAlias = Dict[str, str]


class OpenAIWorkflow(Workflow):
    """(Abstract) Workflow using OpenAI API calls."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        name: str,
        api_key: str,
        model: str,
    ):
        """(Abstract) OpenAI Workflow.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            name: Name of the node (unique identifier)
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
        """
        super().__init__(name=name)
        self._http_client = http_client
        self._client = AsyncOpenAI(http_client=http_client, api_key=api_key)
        self._model = model

    def _log_before(
        self, endpoint: str, context: Context, retry_state: RetryCallState
    ) -> None:
        """Context aware logging before the request is made."""
        if retry_state:
            logger.debug(
                f"Workflow={self.name} calls endpoint={endpoint} within context={context} (Attempt {retry_state.attempt_number})."
            )
        else:
            logger.debug(f"retry_state is {retry_state}; not logging before.")

    def _log_before_sleep(
        self, endpoint: str, context: Context, retry_state: RetryCallState
    ) -> None:
        """Context aware logging before sleeping after a failed request."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f"Attempt {retry_state.attempt_number} of workflow={self.name} "
                f"calling endpoint={endpoint} within context={context} "
                f"failed with error: {retry_state.outcome.exception()}. "
                f"Retrying in {retry_state.upcoming_sleep:.0f} seconds."
            )

    async def _chat_completions_create(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Context,
        **kwargs,
    ) -> ChatCompletion:
        """Calls the OpenAI chat.completions.create endpoint.

        Args:
            context: Logging context for retry logs.
            system_prompt: System prompt for the AI model.
            user_prompt: User prompt for the AI model.
        """
        endpoint = "chat.completions.create"

        # Perform the request and retry if necessary. There is some context aware logging
        #  - `before`: before the request is made (or before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            endpoint=endpoint, context=context, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            endpoint=endpoint, context=context, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **kwargs,
                )
        return response

    async def _chat_completions_parse(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[BaseModel],
        context: Context,
        **kwargs,
    ) -> ParsedChatCompletion:
        """Calls the OpenAI chat.completions.parse endpoint.

        Args:
            system_prompt: System prompt for the AI model.
            user_prompt: User prompt for the AI model.
            response_format: The model into which the response should be parsed.
            context: Logging context for retry logs.
        """
        endpoint = "chat.completions.parse"

        # Perform the request and retry if necessary. There is some context aware logging
        #  - `before`: before the request is made (or before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            endpoint=endpoint, context=context, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            endpoint=endpoint, context=context, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                response = await self._client.chat.completions.parse(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=response_format,  # type: ignore[call-arg]
                    **kwargs,
                )
        return response

    @staticmethod
    def _get_input_param(
        image_url: str,
        system_prompt: str,
        user_prompt: str,
        detail: Literal["low", "high", "auto"],
    ) -> ResponseInputParam:
        # Prepare openai parameters
        image_param: ResponseInputImageParam = {
            "type": "input_image",
            "image_url": image_url,
            "detail": detail,
        }
        input_param: ResponseInputParam = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    image_param,
                ],
            },
        ]
        return input_param

    async def _responses_create(
        self,
        image_url: str,
        system_prompt: str,
        user_prompt: str,
        context: Context,
        **kwargs,
    ) -> Response:
        """Analyses a base64 encoded image.

        Args:
            image_url: Raw base64 encoded image with the data URI scheme.
            system_prompt: System prompt for the AI model.
            user_prompt: User prompt for the AI model.
            context: Logging context for retry logs.

        Note:
            Having the url of a jpeg image (for example), the image_url is optained as:
            ```python
            import requests

            # Read images as bytes
            resp = requests.get(url)
            resp.raise_for_status()
            image = resp.content

            # Encode as base64
            b64 = base64.b64encode(image).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64}"
            ```

            The extracted text can be obtained by `response.output_text`
        """
        # Prepare variables
        endpoint = "response.create"
        detail: Literal["low", "high", "auto"] = "high"
        input_param = self._get_input_param(
            image_url=image_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            detail=detail,
        )

        # Extract information from image
        # Perform the request and retry if necessary. There is some context aware logging
        #  - `before`: before the request is made (or before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            endpoint=endpoint, context=context, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            endpoint=endpoint, context=context, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                response = await self._client.responses.create(
                    model=self._model,
                    input=input_param,
                    **kwargs,
                )
        return response

    async def _responses_parse(
        self,
        image_url: str,
        system_prompt: str,
        user_prompt: str,
        text_format: type[BaseModel],
        context: Context,
        **kwargs,
    ) -> ParsedResponse:
        """Analyses a base64 encoded image and parses the output_text into response_format.

        Args:
            image_url: Raw base64 encoded image with the data URI scheme.
            system_prompt: System prompt for the AI model.
            user_prompt: User prompt for the AI model.
            text_format: The model into which the response should be parsed.
            context: Logging context for retry logs.

        Note:
            (c.f. :func:`_responses_create`)
        """
        # Prepare variables
        endpoint = "response.parse"
        detail: Literal["low", "high", "auto"] = "high"
        input_param = self._get_input_param(
            image_url=image_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            detail=detail,
        )

        # Extract information from image
        # Perform the request and retry if necessary. There is some context aware logging
        #  - `before`: before the request is made (or before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            endpoint=endpoint, context=context, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            endpoint=endpoint, context=context, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                response = await self._client.responses.parse(
                    model=self._model,
                    input=input_param,
                    text_format=text_format,
                    **kwargs,
                )
        return response


class OpenAIClassification(OpenAIWorkflow):
    """Open AI classification workflow with single API call using specific product_item fields for setting up the context.

    Note:
        The system prompt sets the classes to be produced. They must be contained in allowed classes.
        The fields declared in product_item_fields are concatenated for creating a user prompt from
        which the classification should happen.
    """

    _product_prompt_template = "Product Details:\n{product_details}\n\nRelevance:"
    _product_details_template = "{field_name}:\n{field_value}"
    _max_tokens: int = 1

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        name: str,
        api_key: str,
        model: str,
        product_item_fields: List[str],
        system_prompt: str,
        allowed_classes: List[int],
    ):
        """Open AI classification workflow.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            name: Name of the workflow (unique identifier)
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
            product_item_fields: Product item fields used to construct the user prompt.
            system_prompt: System prompt for the AI model.
            allowed_classes: Allowed classes for model output (must be positive).
        """
        super().__init__(
            http_client=http_client,
            name=name,
            api_key=api_key,
            model=model,
        )

        if not self._product_item_fields_are_valid(
            product_item_fields=product_item_fields
        ):
            not_valid_fields = set(product_item_fields) - set(
                ProductItem.model_fields.keys()
            )
            raise ValueError(
                f"Invalid product_item_fields are given: {not_valid_fields}."
            )
        self._product_item_fields = product_item_fields
        self._system_prompt = system_prompt

        if not all(ac >= 0 for ac in allowed_classes):
            raise ValueError("Values of allowed_classes must be >= 0")
        self._allowed_classes = allowed_classes

    @staticmethod
    def _product_item_fields_are_valid(product_item_fields: List[str]) -> bool:
        """Ensure all product_item_fields are valid ProductItem attributes."""
        return set(product_item_fields).issubset(ProductItem.model_fields.keys())

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

    async def _get_product_prompt(self, product: ProductItem) -> str:
        """Forms and returns the product related part for the user_prompt."""

        # Form the product details from the ProductItem
        product_details = self._get_product_details(product=product)
        if not product_details:
            raise ValueError(
                f"Missing product_details for product_item_fields={self._product_item_fields}."
            )

        # Create user prompt
        product_prompt = self._product_prompt_template.format(
            product_details=product_details,
        )
        return product_prompt

    async def _get_user_prompt(self, product: ProductItem) -> str:
        """Forms and returns the user_prompt."""
        product_prompt = await self._get_product_prompt(product=product)
        return product_prompt

    async def _chat_classification(
        self,
        product: ProductItem,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> ClassificationResult:
        """Calls the OpenAI Chat enpoint for a classification."""
        context = {"product.url": product.url}
        response = await self._chat_completions_create(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            **kwargs,
        )

        if (
            not response
            or not (content := response.choices[0].message.content)
            or not (usage := response.usage)
        ):
            raise ValueError(
                f'Error calling OpenAI API: response="{response}, content={content}, usage={usage}".'
            )

        # Convert to ClassificationResult object
        result = int(content.strip())
        return ClassificationResult(
            result=result,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

    async def run(self, product: ProductItem) -> ClassificationResult:
        """Calls the OpenAI API with the user prompt from the product."""

        # Get user prompt
        user_prompt = await self._get_user_prompt(product=product)

        # Call the OpenAI API
        try:
            clfn = await self._chat_classification(
                product=product,
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                max_tokens=self._max_tokens,
            )

            # Enforce that the classification is in the allowed classes
            if clfn.result not in self._allowed_classes:
                raise ValueError(
                    f"classification result={clfn.result} not in allowed_classes={self._allowed_classes}"
                )

        except Exception as e:
            raise Exception(
                f'Error classifying product at url="{product.url}" with workflow="{self.name}": {e}'
            )

        logger.debug(
            f'Classification for url="{product.url}" (workflow={self.name}): result={clfn.result}, tokens used={clfn.input_tokens + clfn.output_tokens}'
        )
        return clfn


class OpenAIClassificationUserInputs(OpenAIClassification):
    """Open AI classification workflow with single API call using specific product_item fields plus user_inputs for setting up the context.

    Note:
        The system prompt sets the classes to be produced. They must be contained in allowed classes.
        The fields declared in product_item_fields together with the user_inputs are concatenated for
        creating a user prompt from which the classification should happen.
    """

    _user_inputs_template = "{key}: {val}"

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        name: str,
        api_key: str,
        model: str,
        product_item_fields: List[str],
        system_prompt: str,
        allowed_classes: List[int],
        user_inputs: UserInputs,
    ):
        """Open AI classification workflow from user input.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            name: Name of the workflow (unique identifier)
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
            product_item_fields: Product item fields used to construct the user prompt.
            system_prompt: System prompt for the AI model.
            allowed_classes: Allowed classes for model output.
            user_inputs: Inputs from the frontend by the user.
        """
        super().__init__(
            http_client=http_client,
            name=name,
            api_key=api_key,
            model=model,
            product_item_fields=product_item_fields,
            system_prompt=system_prompt,
            allowed_classes=allowed_classes,
        )
        user_inputs_strings = [
            self._user_inputs_template.format(key=k, val=v)
            for k, v in user_inputs.items()
        ]
        user_inputs_joined = "\n".join(user_inputs_strings)
        self._user_inputs_prompt = f"User Inputs:\n{user_inputs_joined}"

    async def _get_user_prompt(self, product: ProductItem) -> str:
        """Forms the user_prompt from the product details plus user_inputs."""
        product_prompt = await super()._get_product_prompt(product=product)
        user_prompt = f"{self._user_inputs_prompt}\n\n{product_prompt}"
        return user_prompt
