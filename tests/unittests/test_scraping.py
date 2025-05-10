import pytest

from fraudcrawler.base.base import Setup, Host, Location, Language
from fraudcrawler.scraping.serp import SerpResult
from fraudcrawler import SerpApi, Enricher, ZyteApi
from fraudcrawler.scraping.enrich import Keyword


@pytest.fixture
def serpapi():
    setup = Setup()
    serpapi = SerpApi(api_key=setup.serpapi_key)
    return serpapi


@pytest.fixture
def enricher():
    setup = Setup()
    enricher = Enricher(
        user=setup.dataforseo_user,
        pwd=setup.dataforseo_pwd,
    )
    return enricher


@pytest.fixture
def zyteapi():
    setup = Setup()
    zyteapi = ZyteApi(api_key=setup.zyteapi_key)
    return zyteapi


@pytest.mark.asyncio
async def test_serpapi_search(serpapi):
    search_string = "sildenafil"
    language = Language(name="German")
    location = Location(name="Switzerland")
    num_results = 5
    urls = await serpapi._search(
        search_string=search_string,
        language=language,
        location=location,
        num_results=num_results,
    )
    assert 0 < len(urls) <= num_results
    assert all(isinstance(url, str) for url in urls)
    assert all(url.startswith("http") for url in urls)


def test_serpapi_keep_url(serpapi):
    assert serpapi._keep_url(url="https://example.ch", country_code="ch") is True
    assert serpapi._keep_url(url="https://example.ch/foobar", country_code="ch") is True
    assert serpapi._keep_url(url="https://example.com", country_code="ch") is True
    assert serpapi._keep_url(url="https://example.it", country_code="ch") is False


def test_serpapi_is_excluded(serpapi):
    excluded_urls = [
        Host(name="Springer", domains="springer.com"),
        Host(name="Example", domains="example.com"),
    ]
    assert (
        serpapi._is_excluded(
            domain="example.com",
            excluded_urls=excluded_urls,
        )
        is True
    )
    assert (
        serpapi._is_excluded(
            domain="springer.ch",
            excluded_urls=excluded_urls,
        )
        is False
    )
    assert (
        serpapi._is_excluded(
            domain="link.springer.com",
            excluded_urls=excluded_urls,
        )
        is True
    )
    assert (
        serpapi._is_excluded(
            domain="pringer.com",
            excluded_urls=excluded_urls,
        )
        is False
    )


def test_serpapi_create_serp_result(serpapi):
    url = "https://www.example.ch"
    location = Location(name="Switzerland")
    marketplaces = None
    result = serpapi._create_serp_result(
        url=url,
        location=location,
        marketplaces=marketplaces,
    )
    assert result.url == url
    assert result.domain == "example.ch"
    assert result.marketplace_name == serpapi._default_marketplace_name

    marketplaces = [
        Host(name="Galaxus", domains="galaxus.ch"),
        Host(name="Example", domains="example.ch"),
    ]
    result = serpapi._create_serp_result(
        url=url,
        location=location,
        marketplaces=marketplaces,
    )
    assert result.url == url
    assert result.domain == "example.ch"
    assert result.marketplace_name == "Example"

    marketplaces = [Host(name="Galaxus", domains="galaxus.ch")]
    serp_result = serpapi._create_serp_result(
        url=url,
        location=location,
        marketplaces=marketplaces,
    )
    assert serp_result.url == url
    assert serp_result.domain == "example.ch"
    assert serp_result.marketplace_name == serpapi._default_marketplace_name


@pytest.mark.asyncio
async def test_serpapi_apply_marketplaces(serpapi):
    search_term = "sildenafil"
    language = Language(name="German")
    location = Location(name="Switzerland")
    marketplaces = [Host(name="Ricardo", domains="ricardo.ch")]
    num_results = 5
    results = await serpapi.apply(
        search_term=search_term,
        language=language,
        location=location,
        num_results=num_results,
        marketplaces=marketplaces,
    )
    assert all(isinstance(res, SerpResult) for res in results)
    assert all(res.url.startswith("http") for res in results)


@pytest.mark.asyncio
async def test_serpapi_apply_excluded_urls(serpapi):
    search_term = "sildenafil"
    language = Language(name="German")
    location = Location(name="Switzerland")
    excluded_urls = [Host(name="Altibbi", domains="altibbi.com")]
    num_results = 5
    results = await serpapi.apply(
        search_term=search_term,
        language=language,
        location=location,
        num_results=num_results,
        excluded_urls=excluded_urls,
    )
    assert all(isinstance(res, SerpResult) for res in results)
    assert all(res.url.startswith("http") for res in results)


@pytest.mark.asyncio
async def test_enricher_get_suggested_keywords(enricher):
    search_term = "sildenafil"
    location = Location(name="Switzerland", code="ch")
    language = Language(name="German", code="de")
    limit = 5
    keywords = await enricher._get_suggested_keywords(
        search_term=search_term,
        location=location,
        language=language,
        limit=limit,
    )
    assert 0 < len(keywords) <= limit
    assert all(isinstance(kw, Keyword) for kw in keywords)


@pytest.mark.asyncio
async def test_enricher_get_related_keywords(enricher):
    search_term = "sildenafil"
    location = Location(name="Switzerland", code="ch")
    language = Language(name="German", code="de")
    limit = 5
    keywords = await enricher._get_related_keywords(
        search_term=search_term,
        location=location,
        language=language,
        limit=limit,
    )
    assert 0 < len(keywords) <= limit
    assert all(isinstance(kw, Keyword) for kw in keywords)


@pytest.mark.asyncio
async def test_enricher_apply(enricher):
    search_term = "sildenafil"
    location = Location(name="Switzerland", code="ch")
    language = Language(name="German", code="de")
    n_terms = 5
    terms = await enricher.apply(
        search_term=search_term,
        location=location,
        language=language,
        n_terms=n_terms,
    )
    assert len(terms) == n_terms
    assert search_term not in terms
    assert all(isinstance(t, str) for t in terms)


@pytest.mark.asyncio
async def test_zyteapi_get_details(zyteapi):
    url = "https://www.altibbi.com/answer/159"
    product = await zyteapi.get_details(url=url)
    assert product

    prod_url = product.get("url").replace("://www.", "://")
    url = url.replace("://www.", "://")
    assert prod_url == url
    assert "product" in product
    assert "metadata" in product["product"]


def test_zyteapi_keep_product(zyteapi):
    details = {
        "url": "http://example.ch",
        "product": {
            "name": "sildenafil",
            "description": "buy sildenafil online",
            "metadata": {"probability": 0.5},
        },
    }
    assert zyteapi.keep_product(details=details, threshold=0.1) is True
    assert zyteapi.keep_product(details=details, threshold=0.6) is False
