import pytest
import pytest_asyncio

from fraudcrawler.base.base import (
    Setup,
    Host,
    Location,
    Language,
    HttpxAsyncClient,
)
from fraudcrawler.scraping.search import (
    Search,
    SearchResult,
    SerpAPIGoogle,
    SerpAPIGoogleShopping,
    Toppreise,
)
from fraudcrawler import Enricher, URLCollector, ZyteAPI
from fraudcrawler.scraping.enrich import Keyword


@pytest_asyncio.fixture
async def serpapi_google():
    setup = Setup()
    async with HttpxAsyncClient() as httpx_client:
        yield SerpAPIGoogle(http_client=httpx_client, api_key=setup.serpapi_key)


@pytest_asyncio.fixture
async def serpapi_google_shopping():
    setup = Setup()
    async with HttpxAsyncClient() as httpx_client:
        yield SerpAPIGoogleShopping(http_client=httpx_client, api_key=setup.serpapi_key)


@pytest_asyncio.fixture
async def toppreise():
    async with HttpxAsyncClient() as httpx_client:
        yield Toppreise(http_client=httpx_client)


@pytest_asyncio.fixture
async def search():
    setup = Setup()
    async with HttpxAsyncClient() as httpx_client:
        yield Search(http_client=httpx_client, serpapi_key=setup.serpapi_key, zyte_api=None)


@pytest_asyncio.fixture
async def enricher():
    setup = Setup()
    async with HttpxAsyncClient() as httpx_client:
        yield Enricher(
            http_client=httpx_client,
            user=setup.dataforseo_user,
            pwd=setup.dataforseo_pwd,
        )


@pytest.fixture
def url_collector():
    return URLCollector()


@pytest.fixture
def ricardo_urls():
    """Test URLs from Ricardo with tracking parameters."""
    return [
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/?srsltid=AfmBOor1uTLRhTr9omRJOPPCGfzq0qSwlycUzQVu_w6LYzE3L8y_YL3I",
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/?srsltid=AfmBOorTWSb3cDNoyJjtrdXvma8Uie5RZ7yUf6X9lEL-O1-aFgt5EEjW",
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/?utm_source=google&utm_medium=cpc&srsltid=test",
    ]


@pytest.fixture
def ebay_urls():
    """Test URLs from eBay with tracking parameters."""
    return [
        "https://www.ebay.com/itm/123456?utm_source=test&other_param=value",
        "https://www.ebay.com/itm/789012?srsltid=tracking&utm_campaign=test",
        "https://www.ebay.com/itm/345678?param1=value1&param2=value2",
    ]


@pytest.fixture
def other_urls():
    """Test URLs from other domains with tracking parameters."""
    return [
        "https://www.amazon.com/product/123?utm_source=google&utm_medium=cpc",
        "https://www.galaxus.ch/de/product/456?srsltid=tracking&utm_term=test",
        "https://www.digitec.ch/fr/product/789?utm_campaign=test&other_param=value",
    ]


@pytest_asyncio.fixture
async def zyteapi():
    setup = Setup()
    async with HttpxAsyncClient() as httpx_client:
        yield ZyteAPI(http_client=httpx_client, api_key=setup.zyteapi_key)


@pytest.mark.asyncio
async def test_serpapi_google_search(serpapi_google):
    search_term = "Kaffee"
    language = Language(name="German")
    location = Location(name="Switzerland")
    num_results = 5
    results = await serpapi_google.search(
        search_term=search_term,
        language=language,
        location=location,
        num_results=num_results,
    )
    assert 0 < len(results) <= num_results
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)
    assert all(res.search_engine_name == "google" for res in results)


@pytest.mark.asyncio
async def test_serpapi_google_shopping_search(serpapi_google_shopping):
    search_term = "Kaffee"
    language = Language(name="German")
    location = Location(name="Switzerland")
    num_results = 5
    results = await serpapi_google_shopping.search(
        search_term=search_term,
        language=language,
        location=location,
        num_results=num_results,
    )
    print(f"Results: {results}")
    assert 0 < len(results) <= num_results
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)
    assert all(res.search_engine_name == "google_shopping" for res in results)


def test_search_engine_create_search_result(serpapi_google):
    url = "https://www.example.ch"
    result = serpapi_google._create_search_result(url=url)
    assert isinstance(result, SearchResult)
    assert result.url == url
    assert result.domain == "example.ch"


@pytest.mark.asyncio
async def test_serpapi_google_search_marketplaces(serpapi_google):
    search_term = "Kaffee"
    language = Language(name="German")
    location = Location(name="Switzerland")
    marketplaces = [Host(name="Ricardo", domains="ricardo.ch")]
    num_results = 5
    results = await serpapi_google.search(
        search_term=search_term,
        language=language,
        location=location,
        num_results=num_results,
        marketplaces=marketplaces,
    )
    assert 0 < len(results) <= num_results
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)
    assert all("ricardo.ch" in res.url for res in results)


@pytest.mark.asyncio
async def test_toppreise_search(toppreise):
    search_term = "Liebherr CT 2531"
    num_results = 5
    results = await toppreise.search(
        search_term=search_term,
        num_results=num_results,
    )
    assert 0 < len(results) <= num_results
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)
    assert all(res.search_engine_name == "toppreise" for res in results)


def test_search_apply_filters(search):
    location = Location(name="Switzerland")

    # No filters applied
    result = SearchResult(
        url="https://www.example.ch",
        domain="example.ch",
        search_engine_name="Engine",
    )
    result = search._apply_filters(result=result, location=location)
    assert isinstance(result, SearchResult)
    assert result.url == "https://www.example.ch"
    assert result.domain == "example.ch"
    assert result.search_engine_name == "Engine"
    assert result.filtered is False
    assert result.filtered_at_stage is None

    # Country code filter applied
    result = SearchResult(
        url="https://www.example.org",
        domain="example.org",
        search_engine_name="Engine",
    )
    result = search._apply_filters(result=result, location=location)
    assert isinstance(result, SearchResult)
    assert result.url == "https://www.example.org"
    assert result.domain == "example.org"
    assert result.search_engine_name == "Engine"
    assert result.filtered is True
    assert isinstance(result.filtered_at_stage, str)
    assert result.filtered_at_stage == "Search (country code filtering)"

    # Marketplace filter not applied (would be applied for country code but is overridden)
    result = SearchResult(
        url="https://www.example.org",
        domain="example.org",
        search_engine_name="Engine",
    )
    marketplaces = [Host(name="Example", domains="example.org")]
    result = search._apply_filters(
        result=result, location=location, marketplaces=marketplaces
    )
    assert isinstance(result, SearchResult)
    assert result.url == "https://www.example.org"
    assert result.domain == "example.org"
    assert result.search_engine_name == "Engine"
    assert result.filtered is False
    assert result.filtered_at_stage is None

    # Excluded URLs filter applied
    result = SearchResult(
        url="https://de.example.ch",
        domain="de.example.ch",
        search_engine_name="Engine",
    )
    excluded_urls = [Host(name="Example", domains="example.ch")]
    result = search._apply_filters(
        result=result, location=location, excluded_urls=excluded_urls
    )
    assert isinstance(result, SearchResult)
    assert result.url == "https://de.example.ch"
    assert result.domain == "de.example.ch"
    assert result.search_engine_name == "Engine"
    assert result.filtered is True
    assert isinstance(result.filtered_at_stage, str)
    assert result.filtered_at_stage == "Search (excluded URLs filtering)"

    # No filters applied
    result = SearchResult(
        url="https://www.example.ch",
        domain="example.ch",
        search_engine_name="Engine",
    )
    marketplaces = [Host(name="Example", domains="example.org")]
    excluded_urls = [Host(name="Example", domains="example.de")]
    result = search._apply_filters(
        result=result,
        location=location,
        marketplaces=marketplaces,
        excluded_urls=excluded_urls,
    )
    assert isinstance(result, SearchResult)
    assert result.url == "https://www.example.ch"
    assert result.domain == "example.ch"
    assert result.search_engine_name == "Engine"
    assert result.filtered is False
    assert result.filtered_at_stage is None


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
async def test_enricher_enrich(enricher):
    search_term = "sildenafil"
    location = Location(name="Switzerland", code="ch")
    language = Language(name="German", code="de")
    n_terms = 5
    terms = await enricher.enrich(
        search_term=search_term,
        location=location,
        language=language,
        n_terms=n_terms,
    )
    assert len(terms) == n_terms
    assert search_term not in terms
    assert all(isinstance(t, str) for t in terms)


def test_remove_tracking_parameters_ricardo_urls(url_collector, ricardo_urls):
    """Test that Ricardo URLs are cleaned correctly by removing tracking parameters."""
    expected_clean = (
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/"
    )

    for url in ricardo_urls:
        cleaned = url_collector.remove_tracking_parameters(url)
        assert cleaned == expected_clean, f"Failed to clean URL: {url}"


def test_remove_tracking_parameters_ebay_urls(url_collector, ebay_urls):
    """Test that eBay URLs have all query parameters removed."""
    expected_clean_urls = [
        "https://www.ebay.com/itm/123456",
        "https://www.ebay.com/itm/789012",
        "https://www.ebay.com/itm/345678",
    ]

    for url, expected in zip(ebay_urls, expected_clean_urls):
        cleaned = url_collector.remove_tracking_parameters(url)
        assert cleaned == expected, f"Failed to clean eBay URL: {url}"


def test_remove_tracking_parameters_other_urls(url_collector, other_urls):
    """Test that other domain URLs have tracking parameters removed but keep other params."""
    expected_clean_urls = [
        "https://www.amazon.com/product/123",
        "https://www.galaxus.ch/de/product/456",
        "https://www.digitec.ch/fr/product/789?other_param=value",
    ]

    for url, expected in zip(other_urls, expected_clean_urls):
        cleaned = url_collector.remove_tracking_parameters(url)
        assert cleaned == expected, f"Failed to clean other URL: {url}"


def test_remove_tracking_parameters_no_tracking(url_collector):
    """Test URLs that don't have tracking parameters remain unchanged."""
    clean_urls = [
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/",
        "https://www.ebay.com/itm/123456",
        "https://www.amazon.com/product/123?param1=value1",
    ]

    for url in clean_urls:
        cleaned = url_collector.remove_tracking_parameters(url)
        assert cleaned == url, f"Clean URL was modified: {url}"


def test_remove_tracking_parameters_edge_cases(url_collector):
    """Test edge cases for URL cleaning."""
    test_cases = [
        # URL with only tracking parameters
        (
            "https://www.ricardo.ch/product/?srsltid=test",
            "https://www.ricardo.ch/product/",
        ),
        # URL with mixed tracking and non-tracking parameters
        (
            "https://www.ricardo.ch/product/?param1=value1&srsltid=test&param2=value2",
            "https://www.ricardo.ch/product/?param1=value1&param2=value2",
        ),
        # URL with fragment
        (
            "https://www.ricardo.ch/product/?srsltid=test#section",
            "https://www.ricardo.ch/product/#section",
        ),
        # URL with path parameters
        (
            "https://www.ricardo.ch/product/123/?srsltid=test",
            "https://www.ricardo.ch/product/123/",
        ),
        # Empty URL
        ("", ""),
        # URL without scheme
        ("//www.ricardo.ch/product/?srsltid=test", "//www.ricardo.ch/product/"),
    ]

    for url, expected in test_cases:
        cleaned = url_collector.remove_tracking_parameters(url)
        assert cleaned == expected, f"Failed to clean edge case URL: {url}"


def test_remove_tracking_parameters_known_trackers(url_collector):
    """Test that all known tracking parameters are removed."""
    known_trackers = [
        "srsltid",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
    ]

    for tracker in known_trackers:
        url = f"https://www.ricardo.ch/product/?{tracker}=test_value"
        cleaned = url_collector.remove_tracking_parameters(url)
        assert cleaned == "https://www.ricardo.ch/product/", (
            f"Failed to remove tracker: {tracker}"
        )


@pytest.mark.asyncio
async def test_zyteapi_details(zyteapi):
    url = "https://www.interdiscount.ch/it/product/liebherr-tp1410-136-l-bianco-0005000183"
    product = await zyteapi.details(url=url)
    assert product

    prod_url = product.get("url").replace("://www.", "://")
    url = url.replace("://www.", "://")
    assert prod_url == url
    assert "product" in product
    assert "name" in product["product"]
    assert product["product"]["name"] is not None
    assert "description" in product["product"]
    assert product["product"]["description"] is not None
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


@pytest.mark.asyncio
async def test_search_apply(search):
    search_term = "Kaffee"
    language = Language(name="German")
    location = Location(name="Switzerland")
    num_results = 5
    search_engines = ["google", "google_shopping", "toppreise"]
    results = await search.apply(
        search_term=search_term,
        language=language,
        location=location,
        num_results=num_results,
        search_engines=search_engines,
    )
    assert 0 < len(results) <= len(search_engines) * num_results
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)
