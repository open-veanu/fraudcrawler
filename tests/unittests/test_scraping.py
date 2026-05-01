import os
import uuid
from urllib.parse import parse_qsl, urlparse

import pytest
import pytest_asyncio

from aiocache import Cache
from aiocache.backends.redis import RedisBackend

from fraudcrawler import (
    DistributedURLCollector,
    Enricher,
    LocalURLCollector,
    RedisConfig,
    ZyteAPI,
)
from fraudcrawler.base.base import (
    Setup,
    Host,
    Location,
    Language,
    FilteredAtStage,
    HttpxAsyncClient,
    ProductItem,
    WebsiteSourceMetadata,
)
from fraudcrawler.scraping.enrich import Keyword
from fraudcrawler.scraping.saved_search_models import WebsiteSource
from fraudcrawler.scraping.search import (
    SavedSearchCandidate,
    SavedSearchIngestResult,
    SavedSearchUrlDiagnostic,
    Searcher,
    SearchEngineName,
    SearchResult,
    SerpAPI,
    SerpAPIGoogle,
    SerpAPIGoogleShopping,
    Toppreise,
    WebsiteSearch,
)
from fraudcrawler.settings import ROOT_DIR


def _skip_if_empty_live_results(results, *, context: str) -> None:
    if len(results) == 0:
        pytest.skip(
            f"Live upstream returned zero results ({context}); skipping flaky live assertion."
        )


@pytest_asyncio.fixture
async def serpapi_google():
    setup = Setup()  # type: ignore[call-arg]
    async with HttpxAsyncClient() as httpx_client:
        yield SerpAPIGoogle(http_client=httpx_client, api_key=setup.serpapi_key)


@pytest_asyncio.fixture
async def serpapi_google_shopping():
    setup = Setup()  # type: ignore[call-arg]
    async with HttpxAsyncClient() as httpx_client:
        yield SerpAPIGoogleShopping(http_client=httpx_client, api_key=setup.serpapi_key)


@pytest_asyncio.fixture
async def toppreise():
    setup = Setup()  # type: ignore[call-arg]
    async with HttpxAsyncClient() as httpx_client:
        yield Toppreise(
            http_client=httpx_client,
            zyteapi_key=setup.zyteapi_key,
            redis_use_cache=False,
        )


@pytest_asyncio.fixture
async def searcher():
    setup = Setup()  # type: ignore[call-arg]
    async with HttpxAsyncClient() as httpx_client:
        yield Searcher(
            http_client=httpx_client,
            serpapi_key=setup.serpapi_key,
            zyteapi_key=setup.zyteapi_key,
            redis_use_cache=False,
        )


@pytest_asyncio.fixture
async def enricher():
    setup = Setup()  # type: ignore[call-arg]
    async with HttpxAsyncClient() as httpx_client:
        yield Enricher(
            http_client=httpx_client,
            user=setup.dataforseo_user,
            pwd=setup.dataforseo_pwd,
            redis_use_cache=False,
        )


@pytest.fixture
def url_collector():
    return LocalURLCollector()


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
    setup = Setup()  # type: ignore[call-arg]
    async with HttpxAsyncClient() as httpx_client:
        yield ZyteAPI(
            http_client=httpx_client,
            api_key=setup.zyteapi_key,
            redis_use_cache=False,
        )


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
    _skip_if_empty_live_results(
        results, context="engine=google query=Kaffee location=CH"
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
    assert result.website_source is None


def test_serpapi_get_google_domain():
    """Test the _get_google_domain function for special cases and standard patterns."""
    # Test special case for Brazil
    brazil_location = Location(name="Brazil")
    assert SerpAPI._get_google_domain(brazil_location) == "google.com.br"

    # Test special case for United Kingdom
    uk_location = Location(name="United Kingdom")
    assert SerpAPI._get_google_domain(uk_location) == "google.co.uk"

    # Test special case for Argentina
    argentina_location = Location(name="Argentina")
    assert SerpAPI._get_google_domain(argentina_location) == "google.com.ar"

    # Test standard pattern for Switzerland
    switzerland_location = Location(name="Switzerland")
    assert SerpAPI._get_google_domain(switzerland_location) == "google.ch"

    # Test standard pattern for Germany
    germany_location = Location(name="Germany")
    assert SerpAPI._get_google_domain(germany_location) == "google.de"

    # Test standard pattern for France
    france_location = Location(name="France")
    assert SerpAPI._get_google_domain(france_location) == "google.fr"


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
    language = Language(name="German")
    num_results = 5
    results = await toppreise.search(
        search_term=search_term,
        language=language,
        num_results=num_results,
    )
    assert 0 < len(results) <= num_results
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)
    assert all(res.search_engine_name == "toppreise" for res in results)

    language = Language(name="French")
    results = await toppreise.search(
        search_term=search_term,
        language=language,
        num_results=num_results,
    )
    assert 0 < len(results) <= num_results
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)
    assert all(res.search_engine_name == "toppreise" for res in results)

    language = Language(name="English")
    results = await toppreise.search(
        search_term=search_term,
        language=language,
        num_results=num_results,
    )
    assert 0 < len(results) <= num_results
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)
    assert all(res.search_engine_name == "toppreise" for res in results)


def test_toppreise_get_search_endpoint(toppreise):
    language = Language(name="German", code="de")
    endpoint = toppreise._get_search_endpoint(language=language)
    assert endpoint == "https://www.toppreise.ch/produktsuche"

    language = Language(name="French", code="fr")
    endpoint = toppreise._get_search_endpoint(language=language)
    assert endpoint == "https://www.toppreise.ch/chercher"

    language = Language(name="English", code="en")
    endpoint = toppreise._get_search_endpoint(language=language)
    assert endpoint == "https://www.toppreise.ch/browse"


def test_toppreise_extract_product_urls_from_search_page(toppreise):
    with open(ROOT_DIR / "tests" / "files" / "toppreise_search.html", "rb") as f:
        content = f.read()
    urls = toppreise._extract_product_urls_from_search_page(content=content)
    assert len(urls) == 23
    assert (
        "https://www.toppreise.ch/preisvergleich/Kuehl-Gefrierkombinationen/LIEBHERR-CT-2531-p615781?selsort=rd"
        in urls
    )
    assert (
        "https://www.toppreise.ch/ext_de?pid=0&did=2511&oid=506961161&gdt=MjAyNS0wOS0wNCAyMjo0Mjo0OQ==&slsrt=rd&prcst=shipping&lpos=10"
        in urls
    )


def test_toppreise_extract_product_urls_from_comparison_page(toppreise):
    with open(ROOT_DIR / "tests" / "files" / "toppreise_comparison.html", "rb") as f:
        content = f.read()
    urls = toppreise._extract_product_urls_from_comparison_page(content=content)
    assert len(urls) == 20
    assert (
        "https://www.toppreise.ch/ext_de?pid=615781&did=2532&oid=493842592&gdt=MjAyNS0wOS0wNCAyMjo0NDoyMw==&slsrt=pa&prcst=shipping&lpos=5"
        in urls
    )


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
        cleaned = url_collector._remove_tracking_parameters(url)
        assert cleaned == expected_clean, f"Failed to clean URL: {url}"


def test_remove_tracking_parameters_ebay_urls(url_collector, ebay_urls):
    """Test that eBay URLs have all query parameters removed."""
    expected_clean_urls = [
        "https://www.ebay.com/itm/123456",
        "https://www.ebay.com/itm/789012",
        "https://www.ebay.com/itm/345678",
    ]

    for url, expected in zip(ebay_urls, expected_clean_urls):
        cleaned = url_collector._remove_tracking_parameters(url)
        assert cleaned == expected, f"Failed to clean eBay URL: {url}"


def test_remove_tracking_parameters_other_urls(url_collector, other_urls):
    """Test that other domain URLs have tracking parameters removed but keep other params."""
    expected_clean_urls = [
        "https://www.amazon.com/product/123",
        "https://www.galaxus.ch/de/product/456",
        "https://www.digitec.ch/fr/product/789?other_param=value",
    ]

    for url, expected in zip(other_urls, expected_clean_urls):
        cleaned = url_collector._remove_tracking_parameters(url)
        assert cleaned == expected, f"Failed to clean other URL: {url}"


def test_remove_tracking_parameters_no_tracking(url_collector):
    """Test URLs that don't have tracking parameters remain unchanged."""
    clean_urls = [
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/",
        "https://www.ebay.com/itm/123456",
        "https://www.amazon.com/product/123?param1=value1",
    ]

    for url in clean_urls:
        cleaned = url_collector._remove_tracking_parameters(url)
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
        cleaned = url_collector._remove_tracking_parameters(url)
        assert cleaned == expected, f"Failed to clean edge case URL: {url}"


def _make_product(url: str) -> ProductItem:
    return ProductItem(
        search_term="test",
        search_term_type="seed",
        url=url,
        url_resolved=url,
        search_engine_name="test",
        domain="test.example",
    )


@pytest_asyncio.fixture
async def shared_memory_cache():
    """Provides a per-test aiocache in-memory cache and cleans it up afterwards."""
    namespace = f"test-dedup-{uuid.uuid4().hex}"
    cache = Cache(cache_class=Cache.MEMORY, namespace=namespace)
    yield cache
    await cache.clear()


@pytest_asyncio.fixture
async def live_redis_cache():
    """Per-test live Redis backend; skips the test if Redis is unreachable.

    Entries written during a test are left in Redis for post-mortem
    inspection (e.g. RedisInsight); they self-expire via the dedup TTL
    (`_DUMMY_REDIS_CONFIG.ttl`).

    Env-var overrides:
        REDIS_TEST_HOSTNAME / REDIS_TEST_PORT / REDIS_TEST_DB
            connection params (defaults: localhost:6379, db=15).
        REDIS_TEST_NAMESPACE
            fixed namespace prefix (default: random per-test). Useful to
            point all runs at the same prefix in RedisInsight.
    """
    if fixed_ns := os.environ.get("REDIS_TEST_NAMESPACE"):
        namespace = fixed_ns if fixed_ns.endswith(":") else f"{fixed_ns}:"
    else:
        namespace = f"test-dedup-{uuid.uuid4().hex}:"

    cache = RedisBackend(
        endpoint=os.environ.get("REDIS_TEST_HOSTNAME", "localhost"),
        port=int(os.environ.get("REDIS_TEST_PORT", "6379")),
        db=int(os.environ.get("REDIS_TEST_DB", "15")),
        namespace=namespace,
    )
    try:
        await cache.set(key="__ping__", value="1", ttl=5)
        await cache.delete(key="__ping__")
    except Exception as exc:
        pytest.skip(f"Redis not reachable for live test: {exc}")

    yield cache


_DUMMY_REDIS_CONFIG = RedisConfig(
    hostname="localhost",
    port=6379,
    password=None,
    db=0,
    namespace="test-dedup",
    ttl=60,
)


def _make_dist_collector(id_suffix: str = "") -> DistributedURLCollector:
    """Build a DistributedURLCollector with a dummy config; the Redis backend
    instantiated here is replaced by `_attach_cache` before any I/O."""
    return DistributedURLCollector(
        redis_config=_DUMMY_REDIS_CONFIG, id_suffix=id_suffix
    )


def _attach_cache(collector: DistributedURLCollector, cache) -> DistributedURLCollector:
    collector._cache = cache
    return collector


@pytest.mark.asyncio
async def test_distributed_collector_marks_duplicate_within_instance(
    shared_memory_cache,
):
    collector = _attach_cache(_make_dist_collector(), shared_memory_cache)
    url = "https://www.ricardo.ch/p/123"

    first = await collector.apply(product=_make_product(url))
    assert first.filtered is False
    assert first.filtered_at_stage is None

    second = await collector.apply(product=_make_product(url))
    assert second.filtered is True
    assert second.filtered_at_stage == FilteredAtStage.URL_COLLECTION_CURRENT.value


@pytest.mark.asyncio
async def test_distributed_collector_marks_duplicate_across_instances(
    shared_memory_cache,
):
    url = "https://www.ricardo.ch/p/456"

    first_collector = _attach_cache(_make_dist_collector(), shared_memory_cache)
    await first_collector.apply(product=_make_product(url))

    second_collector = _attach_cache(_make_dist_collector(), shared_memory_cache)
    result = await second_collector.apply(product=_make_product(url))

    assert result.filtered is True
    assert result.filtered_at_stage == FilteredAtStage.URL_COLLECTION_CURRENT.value


@pytest.mark.asyncio
async def test_distributed_collector_different_id_suffix_does_not_share(
    shared_memory_cache,
):
    url = "https://www.ricardo.ch/p/789"

    collector_a = _attach_cache(
        _make_dist_collector(id_suffix="tenant-a"), shared_memory_cache
    )
    await collector_a.apply(product=_make_product(url))

    collector_b = _attach_cache(
        _make_dist_collector(id_suffix="tenant-b"), shared_memory_cache
    )
    result = await collector_b.apply(product=_make_product(url))

    assert result.filtered is False
    assert result.filtered_at_stage is None


@pytest.mark.asyncio
async def test_distributed_collector_add_previously_collected_urls(
    shared_memory_cache,
):
    seed_url = "https://www.ricardo.ch/p/seeded"
    seeded = _attach_cache(_make_dist_collector(), shared_memory_cache)
    await seeded.add_previously_collected_urls(urls=[seed_url])

    fresh = _attach_cache(_make_dist_collector(), shared_memory_cache)
    # Tracking params should be stripped before hashing -> still filtered.
    product = _make_product(f"{seed_url}?utm_source=foo&srsltid=bar")
    result = await fresh.apply(product=product)

    assert result.filtered is True
    assert result.filtered_at_stage == FilteredAtStage.URL_COLLECTION_PREVIOUS.value
    assert result.url == seed_url  # cleaned URL was written back


@pytest.mark.asyncio
async def test_distributed_collector_stores_cleaned_url_marker_in_redis(
    shared_memory_cache,
):
    """First sighting of a tracker-laden URL must persist the CLEANED-URL
    marker into Redis (and never the dirty form)."""
    base_url = "https://www.ricardo.ch/p/with-trackers"
    dirty_url = f"{base_url}?utm_source=foo&srsltid=bar&fbclid=xyz"

    collector = _attach_cache(_make_dist_collector(), shared_memory_cache)
    result = await collector.apply(product=_make_product(dirty_url))

    # First sighting -> not filtered, product.url rewritten to cleaned form
    assert result.filtered is False
    assert result.filtered_at_stage is None
    assert result.url == base_url

    # Cleaned URL's key carries the CURRENT-run marker
    clean_key = collector._get_redis_key(base_url)
    stored = await shared_memory_cache.get(key=clean_key)
    assert stored == FilteredAtStage.URL_COLLECTION_CURRENT.value

    # Dirty URL's key was never written (we never store the tracker form)
    dirty_key = collector._get_redis_key(dirty_url)
    assert await shared_memory_cache.get(key=dirty_key) is None


@pytest.mark.asyncio
async def test_distributed_collector_full_state_machine_on_live_redis(
    live_redis_cache,
):
    """End-to-end dedup against a real Redis instance.

    Skipped when no Redis is reachable. Override host/port/db via
    REDIS_TEST_HOSTNAME / REDIS_TEST_PORT / REDIS_TEST_DB.

    Covers the three legs of the state machine:
      1. seeded URL -> filtered as PREVIOUS
      2. fresh tracker-laden URL -> not filtered, cleaned form persisted
      3. same URL again -> filtered as CURRENT
    """
    seeded_url = "https://www.ricardo.ch/p/seed"
    base_url = "https://www.ricardo.ch/p/fresh"
    dirty_url = f"{base_url}?utm_source=foo&srsltid=bar"

    collector = _attach_cache(_make_dist_collector(), live_redis_cache)

    # 1. PREVIOUS-run marker via explicit seeding
    await collector.add_previously_collected_urls(urls=[seeded_url])
    seeded_result = await collector.apply(product=_make_product(seeded_url))
    assert seeded_result.filtered is True
    assert (
        seeded_result.filtered_at_stage == FilteredAtStage.URL_COLLECTION_PREVIOUS.value
    )

    # 2. First sighting of a dirty URL -> cleaned form persisted as CURRENT
    fresh_result = await collector.apply(product=_make_product(dirty_url))
    assert fresh_result.filtered is False
    assert fresh_result.url == base_url

    clean_key = collector._get_redis_key(base_url)
    stored = await live_redis_cache.get(key=clean_key)
    assert stored == FilteredAtStage.URL_COLLECTION_CURRENT.value

    dirty_key = collector._get_redis_key(dirty_url)
    assert await live_redis_cache.get(key=dirty_key) is None

    # 3. Second sighting -> filtered as CURRENT
    second = await collector.apply(product=_make_product(dirty_url))
    assert second.filtered is True
    assert second.filtered_at_stage == FilteredAtStage.URL_COLLECTION_CURRENT.value


def test_distributed_collector_hash_is_deterministic():
    collector = _make_dist_collector(id_suffix="abc")
    h1 = collector._get_redis_key("https://example.com/p/1")
    h2 = collector._get_redis_key("https://example.com/p/1")
    h3 = collector._get_redis_key("https://example.com/p/2")

    assert h1 == h2
    assert h1 != h3
    domain, _, digest = h1.partition("_")
    assert domain == "example.com"
    assert len(digest) == 64  # sha256 hex digest length


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
        cleaned = url_collector._remove_tracking_parameters(url)
        assert cleaned == "https://www.ricardo.ch/product/", (
            f"Failed to remove tracker: {tracker}"
        )


def test_filter_tracking_query_entries_matches_url_cleaning_behavior(url_collector):
    url = "https://www.ricardo.ch/product/?utm_source=test&param1=value1&srsltid=abc"
    queries = parse_qsl(urlparse(url).query, keep_blank_values=True)

    filtered = url_collector._filter_tracking_query_entries(queries=queries)
    assert filtered == [("param1", "value1")]

    filtered_remove_all = url_collector._filter_tracking_query_entries(
        queries=queries, remove_all=True
    )
    assert filtered_remove_all == []


def test_canonicalize_url_lowercases_host():
    result = WebsiteSearch._canonicalize_url("https://WWW.SHOP.TEST/path")
    assert result == "https://www.shop.test/path"


def test_canonicalize_url_strips_default_ports():
    assert "443" not in WebsiteSearch._canonicalize_url("https://shop.test:443/p")
    assert "80" not in WebsiteSearch._canonicalize_url("http://shop.test:80/p")


def test_canonicalize_url_keeps_non_default_port():
    result = WebsiteSearch._canonicalize_url("https://shop.test:8080/p")
    assert ":8080" in result


def test_canonicalize_url_strips_fragment():
    result = WebsiteSearch._canonicalize_url("https://shop.test/p#section")
    assert "#" not in result


def test_canonicalize_url_removes_tracking_params():
    result = WebsiteSearch._canonicalize_url(
        "https://shop.test/p?color=red&utm_source=google&size=M"
    )
    assert "utm_source" not in result
    assert "color=red" in result
    assert "size=M" in result


def test_canonicalize_url_sorts_query_params():
    result = WebsiteSearch._canonicalize_url("https://shop.test/p?z=1&a=2&q=term")
    assert result == "https://shop.test/p?q=term&a=2&z=1"


@pytest.mark.asyncio
async def test_zyteapi_apply(zyteapi):
    # url = "https://www.interdiscount.ch/it/product/liebherr-tp1410-136-l-bianco-0005000183"
    url = "https://www.toppreise.ch/preisvergleich/Siebtraegermaschinen/DELONGHI-La-Specialista-Maestro-Cold-Brew-EC9885-M-p807974"
    product = await zyteapi.apply(url=url)
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
async def test_searcher_apply(searcher):
    search_term = "Kaffee"
    language = Language(name="German")
    location = Location(name="Switzerland")
    num_results = 5

    # Test with Google
    search_engine = "google"
    results = await searcher.capply(
        search_term=search_term,
        search_engine=search_engine,
        language=language,
        location=location,
        num_results=num_results,
    )
    _skip_if_empty_live_results(
        results, context="searcher.apply engine=google query=Kaffee location=CH"
    )
    assert 0 < len(results)
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)

    # Test with Google Shopping
    search_engine = "google_shopping"
    results = await searcher.capply(
        search_term=search_term,
        search_engine=search_engine,
        language=language,
        location=location,
        num_results=num_results,
    )
    assert 0 < len(results)
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)

    # Test with Toppreise
    search_term = "Liebherr CT 2531"
    search_engine = "toppreise"
    results = await searcher.capply(
        search_term=search_term,
        search_engine=search_engine,
        language=language,
        location=location,
        num_results=num_results,
    )
    assert 0 < len(results)
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)


@pytest.mark.asyncio
async def test_searcher_apply_saved_search_without_source_returns_empty(
    searcher, monkeypatch
):
    language = Language(name="German")
    location = Location(name="Switzerland")

    async def _unexpected_call(**kwargs):  # pragma: no cover - assertion helper
        raise AssertionError(
            "WebsiteSearch.search should not be called without source."
        )

    monkeypatch.setattr(searcher._saved_search_engine, "search", _unexpected_call)

    results = await searcher.capply(
        search_term="Kaffee",
        search_engine=SearchEngineName.WEBSITE_SOURCE,
        language=language,
        location=location,
        num_results=5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_searcher_apply_saved_search_dispatches_via_engine(searcher, monkeypatch):
    language = Language(name="German")
    location = Location(name="Switzerland")
    source = WebsiteSource(
        name="Boost Galaxus",
        urls=[
            {
                "baseUrl": "https://www.galaxus.ch/",
                "searchableUrls": [
                    {
                        "filterUrl": "de/search?q={search_term}",
                        "includeSubstrings": [],
                        "excludeSubstrings": [],
                    }
                ],
            }
        ],
    )

    async def _fake_saved_search(**kwargs):
        assert kwargs["source"] == source
        assert kwargs["search_term"] == "Kaffee"
        assert kwargs["num_results"] == 5
        return [
            SearchResult(
                url="https://www.galaxus.ch/de/product/123",
                domain="galaxus.ch",
                search_engine_name="boost_galaxus_search_engine",
            )
        ]

    async def _fail_post_search(results):  # pragma: no cover - assertion helper
        raise AssertionError(
            "Post-search should not run for saved-search engine results."
        )

    monkeypatch.setattr(searcher._saved_search_engine, "search", _fake_saved_search)
    monkeypatch.setattr(searcher, "_post_search", _fail_post_search)

    results = await searcher.capply(
        search_term="Kaffee",
        search_engine=SearchEngineName.WEBSITE_SOURCE,
        language=language,
        location=location,
        num_results=5,
        website_source_source=source,
    )

    assert len(results) == 1
    assert results[0].search_engine_name == "boost_galaxus_search_engine"
    assert results[0].domain == "galaxus.ch"


@pytest.mark.asyncio
async def test_websitesearch_search_attaches_website_source_metadata(
    searcher, monkeypatch
):
    source = WebsiteSource(
        name="Boost Galaxus",
        urls=[
            {
                "baseUrl": "https://www.galaxus.ch/",
                "searchableUrls": [
                    {
                        "filterUrl": "de/search?q={search_term}",
                        "includeSubstrings": [],
                        "excludeSubstrings": [],
                    }
                ],
            }
        ],
        searchFilterConfig={
            "renderOptions": {
                "javascript": True,
                "includeIframes": False,
                "actions": [],
                "networkCapture": [],
            }
        },
    )

    async def _fake_ingest_source(**kwargs):
        return SavedSearchIngestResult(
            sourceName="Boost Galaxus",
            sourceUrls=["https://www.galaxus.ch/de/search?q=kaffee"],
            candidates=[
                SavedSearchCandidate(
                    url="https://www.galaxus.ch/de/p/coffee-123",
                    title="Coffee 123",
                )
            ],
            urlDiagnostics=[
                SavedSearchUrlDiagnostic(
                    url="de/search?q={search_term}",
                    resolvedUrl="https://www.galaxus.ch/de/search?q=kaffee",
                    renderHttpStatus=200,
                )
            ],
        )

    monkeypatch.setattr(
        searcher._saved_search_engine, "ingest_source", _fake_ingest_source
    )
    results = await searcher._saved_search_engine.search(
        source=source, search_term="kaffee", num_results=5
    )

    assert len(results) == 1
    assert isinstance(results[0].website_source, WebsiteSourceMetadata)
    assert results[0].website_source is not None
    assert results[0].website_source.source_name == "Boost Galaxus"
    assert (
        results[0].website_source.resolved_url
        == "https://www.galaxus.ch/de/search?q=kaffee"
    )
    assert results[0].website_source.render_http_status == 200


@pytest.mark.asyncio
async def test_searcher_apply_accepts_legacy_saved_search_source_keyword(
    searcher, monkeypatch
):
    language = Language(name="German")
    location = Location(name="Switzerland")
    source = WebsiteSource(
        name="Boost Galaxus",
        urls=[
            {
                "baseUrl": "https://www.galaxus.ch/",
                "searchableUrls": [
                    {
                        "filterUrl": "de/search?q={search_term}",
                        "includeSubstrings": [],
                        "excludeSubstrings": [],
                    }
                ],
            }
        ],
    )

    async def _fake_saved_search(**kwargs):
        assert kwargs["source"] == source
        return []

    monkeypatch.setattr(searcher._saved_search_engine, "search", _fake_saved_search)

    results = await searcher.capply(
        search_term="Kaffee",
        search_engine=SearchEngineName.WEBSITE_SOURCE,
        language=language,
        location=location,
        num_results=5,
        saved_search_source=source,
    )

    assert results == []


def test_searcher_apply_filters(searcher):
    location = Location(name="Switzerland")

    # No filters applied
    result = SearchResult(
        url="https://www.example.ch",
        domain="example.ch",
        search_engine_name="Engine",
    )
    result = searcher._apply_filters(result=result, location=location)
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
    result = searcher._apply_filters(result=result, location=location)
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
    result = searcher._apply_filters(
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
    result = searcher._apply_filters(
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
    result = searcher._apply_filters(
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
async def test_searcher_apply_toppreise_post_search(searcher):
    """With the below search term there are links that should be added by post_search."""
    search_term = "Liebherr CT 2531"
    search_engine = "toppreise"
    location = Location(name="Switzerland")
    language = Language(name="German")
    num_results = 16

    results = await searcher.capply(
        search_term=search_term,
        search_engine=search_engine,
        language=language,
        location=location,
        num_results=num_results,
    )
    assert len(results) >= num_results
    assert all(isinstance(res, SearchResult) for res in results)
    assert all(res.url.startswith("http") for res in results)
    assert all(res.search_engine_name == "toppreise" for res in results)
