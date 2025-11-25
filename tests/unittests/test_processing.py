import pytest
import pytest_asyncio
from typing import cast

from fraudcrawler.base.base import Setup, HttpxAsyncClient
from fraudcrawler.processing.processor import ClassificationResult
from fraudcrawler import Processor, ProductItem, OpenAIChat, OpenAIChatUserInputs


@pytest.fixture
def product():
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
async def processor():
    setup = Setup()  # type: ignore[call-arg]
    http_client = HttpxAsyncClient()
    openai_chat = OpenAIChat(
        name="test_openai_chat",
        http_client=http_client,
        api_key=setup.openaiapi_key,
        model="gpt-4o",
        product_item_fields=["product_name", "product_description"],
        system_prompt="You are a random classifier. Choose either 0 or 1. But if it is related to a test, always choose 1.",
        allowed_classes=[0, 1],
    )
    openai_chat_user_input = OpenAIChatUserInputs(
        name="test_openai_chat_user_input",
        http_client=http_client,
        api_key=setup.openaiapi_key,
        model="gpt-4o",
        product_item_fields=["product_name", "product_description"],
        system_prompt="You are a random classifier. Choose either 0 or 1. But if it is related to a test, always choose 1.",
        allowed_classes=[0, 1],
        user_inputs={"one": ["plus", "two"]},
    )
    return Processor(workflows=[openai_chat, openai_chat_user_input])


@pytest.mark.asyncio
async def test_openai_chat_get_product_details(
    processor: Processor, product: ProductItem
):
    openai_chat = cast(OpenAIChat, processor._workflows[0])
    details = openai_chat._get_product_details(product=product)
    assert isinstance(details, str)
    assert "product_name:\nTest Product" in details
    assert "product_description:\nThis is a test product." in details

    openai_chat._product_item_fields = ["not_a_field"]
    details = openai_chat._get_product_details(product)
    assert details == ""


def test_openai_chat_product_item_fields_are_valid():
    assert OpenAIChat._product_item_fields_are_valid(
        ["product_name", "product_description"]
    )
    assert not OpenAIChat._product_item_fields_are_valid(["not_valid_field"])


@pytest.mark.asyncio
async def test_openai_chat_get_user_prompt(processor: Processor, product: ProductItem):
    openai_chat = cast(OpenAIChat, processor._workflows[0])
    product_prompt = await openai_chat._get_user_prompt(product=product)
    assert isinstance(product_prompt, str)
    assert "product_name:\nTest Product" in product_prompt
    assert "product_description:\nThis is a test product." in product_prompt

    openai_chat_user_input = cast(OpenAIChatUserInputs, processor._workflows[1])
    user_prompt = await openai_chat_user_input._get_user_prompt(product=product)
    assert isinstance(user_prompt, str)
    assert product_prompt in user_prompt
    assert "User Inputs:\none: ['plus', 'two']" in user_prompt


@pytest.mark.asyncio
async def test_processor_run(processor: Processor, product: ProductItem):
    classifications = await processor.run(product=product)
    clsfc = classifications["test_openai_chat"]
    assert isinstance(clsfc, ClassificationResult)
    assert isinstance(clsfc.result, int)
    assert isinstance(clsfc.input_tokens, int)
    assert clsfc.input_tokens > 0
    assert isinstance(clsfc.output_tokens, int)
    assert clsfc.output_tokens > 0
