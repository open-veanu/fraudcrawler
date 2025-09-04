import pytest

from fraudcrawler.base.base import (
    Setup,
    Host,
    Location,
    Language,
    Enrichment,
    Deepness,
    Prompt,
    DomainUtils,
    ToppreiseUtils,
)
from fraudcrawler.settings import ROOT_DIR


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


def test_prompt():
    name = "name"
    system_prompt = "this is the system prompt"
    product_item_fields = ["product_name", "product_description"]
    prompt = Prompt(
        name=name,
        system_prompt=system_prompt,
        product_item_fields=product_item_fields,
        allowed_classes=[0, 1],
    )
    assert prompt.name == name
    assert prompt.product_item_fields == product_item_fields
    assert prompt.system_prompt == system_prompt
    assert prompt.allowed_classes == [0, 1]

    with pytest.raises(ValueError):
        Prompt(
            name=name,
            system_prompt=system_prompt,
            product_item_fields=product_item_fields,
            allowed_classes=[-1, 0, 1],
        )

    with pytest.raises(ValueError):
        Prompt(
            name=name,
            system_prompt=system_prompt,
            product_item_fields=product_item_fields,
            allowed_classes=[0.5, 1],
        )


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


def test_toppreise_utils_get_search_endpoint():
    language = Language(name="German", code="de")
    endpoint = ToppreiseUtils._get_search_endpoint(language=language)
    assert endpoint == "https://www.toppreise.ch/produktsuche"

    language = Language(name="French", code="fr")
    endpoint = ToppreiseUtils._get_search_endpoint(language=language)
    assert endpoint == "https://www.toppreise.ch/chercher"

    language = Language(name="English", code="en")
    endpoint = ToppreiseUtils._get_search_endpoint(language=language)
    assert endpoint == "https://www.toppreise.ch/browse"

def test_toppreise_utils_extract_search_product_urls():
    tu = ToppreiseUtils()
    with open(ROOT_DIR / "tests" / "data" / "toppreise_search.html", "rb") as f:
        content = f.read()
    urls = tu._extract_search_product_urls(content=content)
    assert len(urls) == 23
    assert "https://www.toppreise.ch/preisvergleich/Kuehl-Gefrierkombinationen/LIEBHERR-CT-2531-p615781?selsort=rd" in urls
    assert "https://www.toppreise.ch/ext_de?pid=0&did=2511&oid=506961161&gdt=MjAyNS0wOS0wNCAyMjo0Mjo0OQ==&slsrt=rd&prcst=shipping&lpos=10" in urls

def test_toppreise_utils_extract_comparison_product_urls():
    tu = ToppreiseUtils()
    with open(ROOT_DIR / "tests" / "data" / "toppreise_comparison.html", "rb") as f:
        content = f.read()
    urls = tu._extract_comparison_product_urls(content=content)
    assert len(urls) == 20
    assert "https://www.toppreise.ch/ext_de?pid=615781&did=2532&oid=493842592&gdt=MjAyNS0wOS0wNCAyMjo0NDoyMw==&slsrt=pa&prcst=shipping&lpos=5" in urls