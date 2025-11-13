import pytest

from fraudcrawler.base.base import (
    Setup,
    Host,
    Location,
    Language,
    Enrichment,
    Deepness,
    DomainUtils,
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

    host = Host(
        name="Galaxus",
        domains="www.galaxus.ch, https://digitec.ch, https://www.example.com, my.example.com",
    )
    assert host.name == "Galaxus"
    assert host.domains == ["galaxus.ch", "digitec.ch", "example.com", "my.example.com"]


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


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://www.google.com/search?q=x", "google.com"),
        ("https://Google.com", "google.com"),
        ("example.com", "example.com"),
        ("www.example.com", "example.com"),
        ("sub.domain.co.uk", "sub.domain.co.uk"),
        ("https://www.sub.example.co.uk:8080/path", "sub.example.co.uk"),
    ],
)
def test_domain_utils_get_domain(url, expected):
    du = DomainUtils()
    assert du._get_domain(url) == expected
