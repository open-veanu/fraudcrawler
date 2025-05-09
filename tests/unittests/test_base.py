import pytest

from fraudcrawler.base.base import (
    Setup,
    Host,
    Location,
    Language,
    Enrichment,
    Deepness,
    Prompt,
)


def test_setup():
    setup = Setup()
    assert setup.serpapi_key
    assert setup.dataforseo_user
    assert setup.dataforseo_pwd
    assert setup.zyteapi_key
    assert setup.openaiapi_key


def test_host():
    host = Host(name="Galaxus", domains="galaxus.ch, digitec.ch,example.com")
    assert host.name == "Galaxus"
    assert host.domains == ["galaxus.ch", "digitec.ch", "example.com"]

    host = Host(name="Galaxus", domains=["galaxus.ch", "digitec.ch", "example.com"])
    assert host.name == "Galaxus"
    assert host.domains == ["galaxus.ch", "digitec.ch", "example.com"]


def test_location():
    location = Location(name="Switzerland", code="ch")
    assert location.name == "Switzerland"
    assert location.code == "ch"

    location = Location(name="switzerland", code="CH")
    assert location.name == "switzerland"
    assert location.code == "ch"

    location = Location(name="Switzerland")
    assert location.name == "Switzerland"
    assert location.code == "ch"


def test_language():
    language = Language(name="German", code="de")
    assert language.name == "German"
    assert language.code == "de"

    language = Language(name="german", code="DE")
    assert language.name == "german"
    assert language.code == "de"

    language = Language(name="German")
    assert language.name == "German"
    assert language.code == "de"


def test_deepness():
    deepness = Deepness(num_results=20)
    assert deepness.num_results == 20
    assert deepness.enrichment is None

    enrichment = Enrichment(additional_terms=10, additional_urls_per_term=20)
    deepness = Deepness(num_results=20, enrichment=enrichment)
    assert deepness.num_results == 20
    assert deepness.enrichment == enrichment


def test_prompt():
    name = "name"
    context = "this is the context"
    system_prompt = "this is the system prompt"
    prompt = Prompt(
        name=name,
        context=context,
        system_prompt=system_prompt,
        allowed_classes=[0, 1],
    )
    assert prompt.name == name
    assert prompt.context == context
    assert prompt.system_prompt == system_prompt
    assert prompt.allowed_classes == [0, 1]

    with pytest.raises(ValueError):
        Prompt(
            name=name,
            context=context,
            system_prompt=system_prompt,
            allowed_classes=[-1, 0, 1],
        )

    with pytest.raises(ValueError):
        Prompt(
            name=name,
            context=context,
            system_prompt=system_prompt,
            allowed_classes=[0.5, 1],
        )
