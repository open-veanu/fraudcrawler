import logging

from openai import AsyncOpenAI

from fraudcrawler.base.base import Prompt, ClassificationResult
from fraudcrawler.settings import (
    PROCESSOR_USER_PROMPT_TEMPLATE,
    PROCESSOR_DEFAULT_IF_MISSING,
    PROCESSOR_EMPTY_TOKEN_COUNT,
)


logger = logging.getLogger(__name__)


class Processor:
    """Processes product data for classification based on a prompt configuration."""

    def __init__(
        self,
        api_key: str,
        model: str,
        default_if_missing: int = PROCESSOR_DEFAULT_IF_MISSING,
        empty_token_count: int = PROCESSOR_EMPTY_TOKEN_COUNT,
    ):
        """Initializes the Processor.

        Args:
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
            default_if_missing: The default classification to return if error occurs.
            empty_token_count: The default value to return as tokensif the classification is empty.
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._error_response = ClassificationResult(
            result=default_if_missing,
            input_tokens=empty_token_count,
            output_tokens=empty_token_count,
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
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI API")

        # Convert the content to an integer
        content = int(content.strip())

        # For tracking consumption we alre return the tokens used
        classification = ClassificationResult(
            result=content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        return classification

    async def classify(
        self, prompt: Prompt, url: str, product_details: str
    ) -> ClassificationResult:
        """A generic classification method that classifies a product based on a prompt object and returns
          the classification, input tokens, and output tokens.

        Args:
            prompt: A dictionary with keys "system_prompt", etc.
            url: Product URL (often used in the user_prompt).
            product_details: String with product details, formatted per prompt.product_item_fields.

        Note:
            This method returns `PROCESSOR_DEFAULT_IF_MISSING` if:
                - product_details is empty
                - an error occurs during the API call
                - if the response isn't in allowed_classes.
        """
        # If required fields are missing, return the prompt's default fallback if provided.
        if not product_details:
            logger.warning("Missing required product_details for classification.")
            return self._error_response

        # Substitute placeholders in user_prompt with the relevant arguments
        user_prompt = PROCESSOR_USER_PROMPT_TEMPLATE.format(
            product_details=product_details,
        )

        # Call the OpenAI API
        try:
            logger.debug(
                f'Calling OpenAI API for classification (url="{url}", prompt="{prompt.name}")'
            )
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
