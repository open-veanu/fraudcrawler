import base64
import json
from pydantic import BaseModel
import pytest
import pytest_asyncio
from typing import cast, List

from fraudcrawler.settings import ROOT_DIR
from fraudcrawler.base.base import Setup, HttpxAsyncClient
from fraudcrawler.processing.base import ClassificationResult
from fraudcrawler import (
    Processor,
    ProductItem,
    OpenAIClassification,
    OpenAIClassificationUserInputs,
)


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
    openai_clfc = OpenAIClassification(
        name="test_openai_clfc",
        http_client=http_client,
        api_key=setup.openaiapi_key,
        model="gpt-4o",
        product_item_fields=["product_name", "product_description"],
        system_prompt="You are a random classifier. Choose either 0 or 1. But if it is related to a test, always choose 1.",
        allowed_classes=[0, 1],
    )
    openai_clfc_user_input = OpenAIClassificationUserInputs(
        name="test_openai_clfc_user_input",
        http_client=http_client,
        api_key=setup.openaiapi_key,
        model="gpt-4o",
        product_item_fields=["product_name", "product_description"],
        system_prompt="You are a random classifier. Choose either 0 or 1. But if it is related to a test, always choose 1.",
        allowed_classes=[0, 1],
        user_inputs={"one": ["plus", "two"]},
    )
    return Processor(workflows=[openai_clfc, openai_clfc_user_input])


@pytest.mark.asyncio
async def test_openai_clfc_get_product_details(
    processor: Processor, product: ProductItem
):
    openai_clfc = cast(OpenAIClassification, processor._workflows[0])
    details = openai_clfc._get_product_details(product=product)
    assert isinstance(details, str)
    assert "product_name:\nTest Product" in details
    assert "product_description:\nThis is a test product." in details

    openai_clfc._product_item_fields = ["not_a_field"]
    details = openai_clfc._get_product_details(product)
    assert details == ""


def test_openai_clfc_product_item_fields_are_valid():
    assert OpenAIClassification._product_item_fields_are_valid(
        ["product_name", "product_description"]
    )
    assert not OpenAIClassification._product_item_fields_are_valid(["not_valid_field"])


@pytest.mark.asyncio
async def test_openai_clfc_get_user_prompt(processor: Processor, product: ProductItem):
    openai_clfc = cast(OpenAIClassification, processor._workflows[0])
    product_prompt = await openai_clfc._get_user_prompt(product=product)
    assert isinstance(product_prompt, str)
    assert "product_name:\nTest Product" in product_prompt
    assert "product_description:\nThis is a test product." in product_prompt

    openai_clfc_user_input = cast(
        OpenAIClassificationUserInputs, processor._workflows[1]
    )
    user_prompt = await openai_clfc_user_input._get_user_prompt(product=product)
    assert isinstance(user_prompt, str)
    assert product_prompt in user_prompt
    assert "User Inputs:\none: ['plus', 'two']" in user_prompt


@pytest.mark.asyncio
async def test_openai_classification_run(processor: Processor, product: ProductItem):
    openai_clfc = cast(OpenAIClassification, processor._workflows[0])
    clfc = await openai_clfc.run(product=product)

    assert isinstance(clfc, ClassificationResult)
    assert isinstance(clfc.result, int)
    assert isinstance(clfc.input_tokens, int)
    assert clfc.input_tokens > 0
    assert isinstance(clfc.output_tokens, int)
    assert clfc.output_tokens > 0


@pytest.mark.asyncio
async def test_openai_chat_responses_parse(processor: Processor):
    class Test(BaseModel):
        first_name: str
        last_name: str

    openai_clfc = cast(OpenAIClassification, processor._workflows[0])
    response = await openai_clfc._chat_completions_parse(
        system_prompt="You are a expert in doing anything.",
        user_prompt="Parse the following sentence: My first_name is FOO, and last_name is BAR",
        response_format=Test,
        context={"test": "me!"},
    )
    output = response.choices[0].message.parsed
    assert isinstance(output, Test)
    assert output.first_name == "FOO"
    assert output.last_name == "BAR"


@pytest.mark.asyncio
async def test_openai_responses_create(processor: Processor):
    with open(ROOT_DIR / "tests" / "files" / "image.jpg", "rb") as f:
        content = f.read()

    b64 = base64.b64encode(content).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{b64}"

    openai_clfc = cast(OpenAIClassification, processor._workflows[0])
    response = await openai_clfc._responses_create(
        image_url=image_url,
        system_prompt="You are a expert image text extractor.",
        user_prompt="Extract the text of the following image",
        context={"test": "me!"},
    )
    output_text = response.output_text
    assert isinstance(output_text, str)
    assert "CASO Design" in output_text
    assert "791" in output_text


@pytest.mark.asyncio
async def test_openai_responses_parse(processor: Processor, product: ProductItem):
    class ImageText(BaseModel):
        manufacturer_product_name: str
        manufacturer_product_number: str
        colored_arrow_scale_texts: List[str]
        efficiency_class_indicator_arrow_text: str

    with open(ROOT_DIR / "tests" / "files" / "image.jpg", "rb") as f:
        content = f.read()

    b64 = base64.b64encode(content).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{b64}"

    openai_clfc = cast(OpenAIClassification, processor._workflows[0])
    response = await openai_clfc._responses_parse(
        image_url=image_url,
        system_prompt="You are a expert image text extractor.",
        user_prompt="Extract the text of the following image",
        text_format=ImageText,
        context={"test": "me!"},
    )
    output_text = response.output_text
    output = ImageText(**json.loads(output_text))
    assert output.manufacturer_product_name == "CASO Design"
    assert output.manufacturer_product_number == "791"
    assert set(output.colored_arrow_scale_texts) == {"A", "B", "C", "D", "E", "F", "G"}
    assert output.efficiency_class_indicator_arrow_text == "F"


@pytest.mark.asyncio
async def test_processor_run(processor: Processor, product: ProductItem):
    results = await processor.run(product=product)
    clfc = cast(ClassificationResult, results["test_openai_clfc"])
    assert isinstance(clfc.result, int)
    assert isinstance(clfc.input_tokens, int)
    assert clfc.input_tokens > 0
    assert isinstance(clfc.output_tokens, int)
    assert clfc.output_tokens > 0
