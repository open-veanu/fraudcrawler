import pytest

from fraudcrawler.settings import (
    PROCESSOR_DEFAULT_MODEL,
    PROCESSOR_DEFAULT_IF_MISSING,
)

from fraudcrawler.base.base import Setup, ClassificationResult
from fraudcrawler import Processor, Prompt, ProductItem


@pytest.fixture
def product_item():
    return ProductItem(
        search_term="test product",
        search_term_type="original",
        url="https://example.com",
        marketplace_name="Example Marketplace",
        domain="example.com",
        product_name="Test Product",
        product_description="This is a test product.",
        product_price="9.99",
    )


@pytest.fixture
def prompt():
    return Prompt(
        name="test_prompt",
        product_item_fields=["product_name", "product_description"],
        system_prompt="You are a random classifier. Choose either 0 or 1.",
        allowed_classes=[0, 1],
    )


@pytest.fixture
def processor():
    setup = Setup()
    processor = Processor(api_key=setup.openaiapi_key, model=PROCESSOR_DEFAULT_MODEL)
    return processor


def test_processor_get_product_details(product_item, prompt):
    details = Processor._get_product_details(product_item, prompt)
    assert isinstance(details, str)
    assert "product_name:\nTest Product" in details
    assert "product_description:\nThis is a test product." in details

    prompt.product_item_fields = ["not_a_field"]
    details = Processor._get_product_details(product_item, prompt)
    assert details == ""


@pytest.mark.asyncio
async def test_processor_classify_product(processor, product_item, prompt):
    classification = await processor.classify(
        product=product_item,
        prompt=prompt,
    )
    assert isinstance(classification, ClassificationResult)
    assert isinstance(classification.result, int)
    assert isinstance(classification.input_tokens, int)
    assert isinstance(classification.output_tokens, int)
    assert (
        classification.result in prompt.allowed_classes
        or classification.result == PROCESSOR_DEFAULT_IF_MISSING
    )
