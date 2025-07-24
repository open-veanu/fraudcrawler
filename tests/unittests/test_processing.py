import pytest

from fraudcrawler.settings import (
    PROCESSOR_DEFAULT_MODEL,
    PROCESSOR_DEFAULT_IF_MISSING,
)

from fraudcrawler.base.base import Setup, ClassificationResult
from fraudcrawler import Processor, Prompt


@pytest.fixture
def processor():
    setup = Setup()
    processor = Processor(api_key=setup.openaiapi_key, model=PROCESSOR_DEFAULT_MODEL)
    return processor


@pytest.mark.asyncio
async def test_processor_classify_product(processor):
    product_item_fields = ["product_name", "product_description"]
    system_prompt = "You are a specialist for medical products."
    allowed_classes = [0, 1]
    prompt = Prompt(
        name="test_prompt",
        product_item_fields=product_item_fields,
        system_prompt=system_prompt,
        allowed_classes=allowed_classes,
    )
    product_details = ["product_name", "product_description"]
    classification = await processor.classify(
        prompt=prompt,
        url="https://example.com",
        product_details=product_details,
    )
    assert isinstance(classification, ClassificationResult)
    assert isinstance(classification.result, int)
    assert isinstance(classification.input_tokens, int)
    assert isinstance(classification.output_tokens, int)
    assert (
        classification.result in allowed_classes
        or classification.result == PROCESSOR_DEFAULT_IF_MISSING
    )
