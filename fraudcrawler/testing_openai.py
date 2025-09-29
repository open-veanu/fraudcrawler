#!/usr/bin/env python3
"""
Testing OpenAI module for evaluating relevance from ingested text.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from openai import AsyncOpenAI
from tenacity import RetryCallState

from fraudcrawler.base.base import ProductItem
from fraudcrawler.base.retry import get_async_retry
from fraudcrawler.settings import (
    PROCESSOR_DEFAULT_MODEL,
    PROCESSOR_DEFAULT_IF_MISSING,
    PROCESSOR_EMPTY_TOKEN_COUNT,
)

logger = logging.getLogger(__name__)


@dataclass
class RelevanceEvaluationResult:
    """Result of a relevance evaluation."""
    
    relevance_score: float  # Score between 0.0 and 1.0
    confidence: float       # Confidence in the evaluation
    reasoning: str          # Explanation for the score
    input_tokens: int       # Number of input tokens used
    output_tokens: int      # Number of output tokens used
    model_used: str         # Model used for evaluation


class RelevanceEvaluator:
    """Evaluates relevance of text content using OpenAI models."""
    
    def __init__(
        self,
        api_key: str,
        model: str = PROCESSOR_DEFAULT_MODEL,
    ):
        """Initialize the RelevanceEvaluator."""
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def evaluate_relevance(
        self,
        text: str,
        search_term: str,
        context: Optional[str] = None,
        max_tokens: int = 500,
    ) -> RelevanceEvaluationResult:
        """Evaluate the relevance of text content to a search term."""
        
        system_prompt = (
            "You are an expert content evaluator. Evaluate how relevant a given text is "
            "to a specific search term. Provide:\n"
            "1. A relevance score between 0.0 and 1.0 (1.0 = highly relevant)\n"
            "2. A confidence score between 0.0 and 1.0\n"
            "3. A brief reasoning for your evaluation\n\n"
            "Respond in JSON format:\n"
            "{\n"
            '  "relevance_score": 0.85,\n'
            '  "confidence": 0.9,\n'
            '  "reasoning": "This text directly discusses the search term."\n'
            "}"
        )

        user_prompt = f"Search Term: {search_term}\n\nText to Evaluate:\n{text}"
        if context:
            user_prompt += f"\n\nAdditional Context:\n{context}"
        user_prompt += "\n\nPlease evaluate the relevance of the text to the search term."

        logger.info(f"Evaluating relevance for search term '{search_term}'")

        try:
            # Call OpenAI API with retry logic
            retry = get_async_retry()
            async for attempt in retry:
                with attempt:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                    )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI API")

            # Parse JSON response
            try:
                response_data = json.loads(content)
                relevance_score = float(response_data.get("relevance_score", 0.0))
                confidence = float(response_data.get("confidence", 0.0))
                reasoning = str(response_data.get("reasoning", "No reasoning provided"))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse OpenAI response: {e}")
                relevance_score = 0.5
                confidence = 0.5
                reasoning = f"Failed to parse response: {content}"

            # Validate scores
            relevance_score = max(0.0, min(1.0, relevance_score))
            confidence = max(0.0, min(1.0, confidence))

            return RelevanceEvaluationResult(
                relevance_score=relevance_score,
                confidence=confidence,
                reasoning=reasoning,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model_used=self._model,
            )

        except Exception as e:
            logger.error(f"Error evaluating relevance: {e}")
            return RelevanceEvaluationResult(
                relevance_score=0.5,
                confidence=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                input_tokens=PROCESSOR_EMPTY_TOKEN_COUNT,
                output_tokens=PROCESSOR_EMPTY_TOKEN_COUNT,
                model_used=self._model,
            )

    async def evaluate_product_relevance(
        self,
        product: ProductItem,
        search_term: str,
    ) -> RelevanceEvaluationResult:
        """Evaluate the relevance of a ProductItem to a search term."""
        
        # Extract text content from the product
        text_parts = []
        
        if product.product_name:
            text_parts.append(f"Product Name: {product.product_name}")
        
        if product.product_description:
            text_parts.append(f"Description: {product.product_description}")
        
        if product.html_clean:
            text_parts.append(f"Content: {product.html_clean}")
        
        if not text_parts:
            logger.warning(f"No text content available for product at {product.url}")
            return RelevanceEvaluationResult(
                relevance_score=0.0,
                confidence=0.0,
                reasoning="No text content available for evaluation",
                input_tokens=0,
                output_tokens=0,
                model_used=self._model,
            )

        # Combine text parts
        combined_text = "\n\n".join(text_parts)
        context = f"Product URL: {product.url}\nDomain: {product.domain}"
        
        return await self.evaluate_relevance(
            text=combined_text,
            search_term=search_term,
            context=context,
        )


# Convenience function
async def evaluate_text_relevance(
    text: str,
    search_term: str,
    api_key: str,
    model: str = PROCESSOR_DEFAULT_MODEL,
    context: Optional[str] = None,
) -> RelevanceEvaluationResult:
    """Convenience function to quickly evaluate text relevance."""
    evaluator = RelevanceEvaluator(api_key=api_key, model=model)
    return await evaluator.evaluate_relevance(text, search_term, context)


if __name__ == "__main__":
    async def example_usage():
        """Example usage of the relevance evaluator."""
        api_key = "your-openai-api-key-here"
        
        evaluator = RelevanceEvaluator(api_key=api_key)
        
        text = "This is a high-quality food storage container made of durable plastic."
        search_term = "food containers"
        
        result = await evaluator.evaluate_relevance(text, search_term)
        
        print(f"Relevance Score: {result.relevance_score:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Tokens Used: {result.input_tokens + result.output_tokens}")
        print(f"Model: {result.model_used}")
    
    asyncio.run(example_usage())

