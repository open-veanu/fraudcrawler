import pytest

from fraudcrawler.settings import (
    PROCESSOR_DEFAULT_MODEL,
    PROCESSOR_DEFAULT_IF_MISSING,
)

from fraudcrawler.base.base import Setup
from fraudcrawler import Processor, Prompt


@pytest.fixture
def processor():
    setup = Setup()
    processor = Processor(api_key=setup.openaiapi_key, model=PROCESSOR_DEFAULT_MODEL)
    return processor


@pytest.mark.asyncio
async def test_processor_classify_product(processor):
    context = "We are interested in medical products"
    system_prompt = "You are a specialist for medical products."
    allowed_classes = [0, 1]
    prompt = Prompt(
        name="test_prompt",
        context=context,
        system_prompt=system_prompt,
        allowed_classes=allowed_classes,
    )
    name = "sildenafil"
    description = "buy sildenafil online"
    classification = await processor.classify(
        prompt=prompt,
        url="https://example.com",
        name=name,
        description=description,
    )
    assert isinstance(classification, int)
    assert (
        classification in allowed_classes or classification == PROCESSOR_DEFAULT_IF_MISSING
    )
