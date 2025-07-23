import pytest

from fraudcrawler.base.base import Setup, Host, Location, Language
from fraudcrawler.scraping.serp import SerpResult
from fraudcrawler import SerpApi, SearchEngine, Enricher, URLCollector, ZyteApi
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


@pytest.fixture
def zyteapi():
    setup = Setup()
    zyteapi = ZyteApi(api_key=setup.zyteapi_key)
    return zyteapi


@pytest.mark.asyncio
async def test_serpapi_google_search(serpapi):
    search_string = "Kaffee"
    language = Language(name="German")
    location = Location(name="Switzerland")
    num_results = 5
    results = await serpapi._search_google(
        search_string=search_string,
        language=language,
        location=location,
        num_results=num_results,
    )
    assert 0 < len(results) <= num_results
    assert all(isinstance(res, SerpResult) for res in results)
    assert all(res.url.startswith("http") for res in results)


@pytest.mark.asyncio
async def test_serpapi_google_shopping_search(serpapi):
    search_string = "Kaffee"
    language = Language(name="German")
    location = Location(name="Switzerland")
    num_results = 5
    results = await serpapi._search_google_shopping(
        search_string=search_string,
        language=language,
        location=location,
        num_results=num_results,
    )
    assert 0 < len(results) <= num_results
    assert all(isinstance(res, SerpResult) for res in results)
    assert all(res.url.startswith("http") for res in results)


def test_serpapi_apply_filters(serpapi):
    location = Location(name="Switzerland")

    # No filters applied
    result = SerpResult(
        url="https://www.example.ch",
        domain="example.ch",
        marketplace_name="Example",
    )
    result = serpapi._apply_filters(result=result, location=location)
    assert isinstance(result, SerpResult)
    assert result.url == "https://www.example.ch"
    assert result.domain == "example.ch"
    assert result.marketplace_name == "Example"
    assert result.filtered is False
    assert result.filtered_at_stage is None

    # Country code filter applied
    result = SerpResult(
        url="https://www.example.org",
        domain="example.org",
        marketplace_name="Example",
    )
    result = serpapi._apply_filters(result=result, location=location)
    assert isinstance(result, SerpResult)
    assert result.url == "https://www.example.org"
    assert result.domain == "example.org"
    assert result.marketplace_name == "Example"
    assert result.filtered is True
    assert isinstance(result.filtered_at_stage, str)
    assert result.filtered_at_stage == "SerpAPI (country code filtering)"

    # Marketplace filter not applied (would be applied for country code but is overridden)
    result = SerpResult(
        url="https://www.example.org",
        domain="example.org",
        marketplace_name="Example",
    )
    marketplaces = [Host(name="Example", domains="example.org")]
    result = serpapi._apply_filters(
        result=result, location=location, marketplaces=marketplaces
    )
    assert isinstance(result, SerpResult)
    assert result.url == "https://www.example.org"
    assert result.domain == "example.org"
    assert result.marketplace_name == "Example"
    assert result.filtered is False
    assert result.filtered_at_stage is None

    # Excluded URLs filter applied
    result = SerpResult(
        url="https://de.example.ch",
        domain="de.example.ch",
        marketplace_name="Example",
    )
    excluded_urls = [Host(name="Example", domains="example.ch")]
    result = serpapi._apply_filters(
        result=result, location=location, excluded_urls=excluded_urls
    )
    assert isinstance(result, SerpResult)
    assert result.url == "https://de.example.ch"
    assert result.domain == "de.example.ch"
    assert result.marketplace_name == "Example"
    assert result.filtered is True
    assert isinstance(result.filtered_at_stage, str)
    assert result.filtered_at_stage == "SerpAPI (excluded URLs filtering)"

    # No filters applied
    result = SerpResult(
        url="https://www.example.ch",
        domain="example.ch",
        marketplace_name="Example",
    )
    marketplaces = [Host(name="Example", domains="example.org")]
    excluded_urls = [Host(name="Example", domains="example.de")]
    result = serpapi._apply_filters(
        result=result,
        location=location,
        marketplaces=marketplaces,
        excluded_urls=excluded_urls,
    )
    assert isinstance(result, SerpResult)
    assert result.url == "https://www.example.ch"
    assert result.domain == "example.ch"
    assert result.marketplace_name == "Example"
    assert result.filtered is False
    assert result.filtered_at_stage is None


def test_serpapi_create_serp_result(serpapi):
    engine = "google"
    url = "https://www.example.ch"
    location = Location(name="Switzerland")
    result = serpapi._create_serp_result(
        engine=engine,
        url=url,
        location=location,
    )
    assert isinstance(result, SerpResult)
    assert result.url == url
    assert result.domain == "example.ch"
    assert result.marketplace_name == "Google"

    marketplaces = [
        Host(name="Galaxus", domains="galaxus.ch"),
        Host(name="Example", domains="example.ch"),
    ]
    result = serpapi._create_serp_result(
        engine=engine,
        url=url,
        location=location,
        marketplaces=marketplaces,
    )
    assert isinstance(result, SerpResult)
    assert result.url == url
    assert result.domain == "example.ch"
    assert result.marketplace_name == "Example"

    marketplaces = [Host(name="Galaxus", domains="galaxus.ch")]
    serp_result = serpapi._create_serp_result(
        engine=engine,
        url=url,
        location=location,
        marketplaces=marketplaces,
    )
    assert isinstance(serp_result, SerpResult)
    assert serp_result.url == url
    assert serp_result.domain == "example.ch"
    assert serp_result.marketplace_name == "Google"


@pytest.mark.asyncio
async def test_serpapi_apply_marketplaces(serpapi):
    search_term = "sildenafil"
    language = Language(name="German")
    location = Location(name="Switzerland")
    marketplaces = [Host(name="Ricardo", domains="ricardo.ch")]
    num_results = 5
    results = await serpapi.apply(
        search_term=search_term,
        search_engines=[SearchEngine.GOOGLE],
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
        search_engines=[SearchEngine.GOOGLE, SearchEngine.GOOGLE_SHOPPING],
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
