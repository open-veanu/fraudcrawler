from collections import defaultdict
import pytest
import pytest_asyncio
from typing import Any, Dict

from fraudcrawler.settings import PROCESSOR_DEFAULT_IF_MISSING
from fraudcrawler.base.base import Setup, HttpxAsyncClient
from fraudcrawler.processing.processor import ClassificationResult
from fraudcrawler import Processor, ProductItem, OpenAIChat


@pytest.fixture
def product_item():
    return ProductItem(
        search_term="test product",
        search_term_type="original",
        url="https://example.com",
        url_resolved="https://example.com/resolved",
        search_engine_name="Search Engine",
        domain="example.com",
        product_name="Test Product",
        product_description="This is a test product.",
        product_price="9.99",
    )


@pytest_asyncio.fixture
async def openai_chat():
    setup = Setup()
    async with HttpxAsyncClient() as http_client:
        yield OpenAIChat(
            name='test_my_chat',
            http_client=http_client,
            api_key=setup.openaiapi_key,
            model="gpt-4o",
            product_item_fields=['product_name', 'product_description'],
            system_prompt="You are a random classifier. Choose either 0 or 1. But if it is related to a test, always choose 1.",
            allowed_classes=[0, 1],
        )


@pytest.mark.asyncio
async def test_openai_chat_get_product_details(openai_chat, product_item):
    details = openai_chat._get_product_details(product_item)
    assert isinstance(details, str)
    assert "product_name:\nTest Product" in details
    assert "product_description:\nThis is a test product." in details

    openai_chat._product_item_fields = ["not_a_field"]
    details = openai_chat._get_product_details(product_item)
    assert details == ""


@pytest.mark.asyncio
async def test_processor_run(openai_chat, product_item):
    processor = Processor(workflows=[openai_chat])
    classification = await processor.run(product=product_item)
    classification = classification[openai_chat.name]
    print()
    print(classification)
    print()
    assert isinstance(classification, ClassificationResult)
    assert isinstance(classification.result, int)
    assert isinstance(classification.input_tokens, int)
    assert classification.input_tokens > 0
    assert isinstance(classification.output_tokens, int)
    assert classification.output_tokens > 0
    assert (
        classification.result in openai_chat._allowed_classes
        or classification.result == PROCESSOR_DEFAULT_IF_MISSING
    )
