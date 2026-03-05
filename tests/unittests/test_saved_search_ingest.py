from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pydantic import ValidationError

from fraudcrawler.scraping.saved_search_extraction import (
    extract_candidate_urls_from_render_captures,
)
from fraudcrawler.scraping.saved_search_filters import (
    apply_candidate_url_pattern_filters,
)
from fraudcrawler.scraping.saved_search_ingest import (
    SavedSearchIngestService,
    canonicalize_url,
    merge_candidates_with_precedence,
    normalize_url,
    should_trigger_render_fallback,
)
from fraudcrawler.scraping.saved_search_models import (
    SavedSearchCandidate,
    SavedSearchRenderedNetworkCapture,
    SavedSearchSource,
)


def test_old_graphql_source_fields_are_rejected():
    with pytest.raises(ValidationError):
        SavedSearchSource(
            name="invalid-source",
            urls=[{"rawUrl": "https://shop.test/list"}],
            useGraph=True,
            graphURL="https://shop.test/graphql",
        )


def test_old_graphql_filter_config_field_is_rejected():
    with pytest.raises(ValidationError):
        SavedSearchSource(
            name="invalid-source",
            urls=[{"rawUrl": "https://shop.test/list"}],
            searchFilterConfig={"graphql": {"enabled": True}},
        )


def test_old_filter_entry_id_and_key_fields_are_rejected():
    with pytest.raises(ValidationError):
        SavedSearchSource(
            name="invalid-filter-entry",
            urls=[{"rawUrl": "https://shop.test/list"}],
            searchFilterConfig={
                "entries": [
                    {
                        "id": "legacy-id",
                        "domain": "generic",
                        "key": "filter",
                        "value": "100=1",
                        "enabled": True,
                    }
                ]
            },
        )


def test_include_exclude_filters_include_all_and_exclude_any():
    candidates = [
        SavedSearchCandidate(url="https://shop.test/s1/product/a-123", title="A"),
        SavedSearchCandidate(url="https://shop.test/s1/category/a-123", title="B"),
        SavedSearchCandidate(url="https://shop.test/s1/product/blocked-123", title="C"),
    ]
    source = SavedSearchSource(
        name="test",
        urls=[{"rawUrl": "https://shop.test/list"}],
        searchFilterConfig={
            "candidateUrlIncludeSubstrings": ["/s1/product/"],
            "candidateUrlExcludeSubstrings": ["BLOCKED"],
        },
    )
    result = apply_candidate_url_pattern_filters(
        candidates, source.search_filter_config
    )
    assert [item.url for item in result.filtered_candidates] == [
        "https://shop.test/s1/product/a-123"
    ]
    assert result.dropped_by_missing_include_all_match == 1
    assert result.dropped_by_exclude_substring == 1


@pytest.mark.parametrize(
    "policy,status,count,expected",
    [
        ("always", None, 3, True),
        ("always", 403, 0, True),
        ("on_http_403", 403, 2, True),
        ("on_http_403", 200, 0, False),
        ("on_zero_candidates", 200, 0, True),
        ("on_zero_candidates", 403, 1, False),
    ],
)
def test_fallback_trigger_matrix(policy, status, count, expected):
    assert should_trigger_render_fallback(policy, status, count) is expected


def test_normalize_url_handles_relative_and_strips_trackers():
    normalized = normalize_url(
        base_url="https://shop.test/catalog/list?q=fridge&utm_source=newsletter",
        href="../p/item-123?utm_medium=email&x=1#details",
    )
    assert normalized == "https://shop.test/p/item-123?x=1"


def test_normalize_url_handles_absolute_href_with_canonical_order():
    normalized = normalize_url(
        base_url="https://shop.test/catalog/list?q=fridge",
        href="https://shop.test/p/item-123?z=9&utm_source=newsletter&q=abc",
    )
    assert normalized == "https://shop.test/p/item-123?q=abc&z=9"


def test_network_capture_parsing_extracts_urls_and_fields():
    captures = [
        SavedSearchRenderedNetworkCapture(
            url="https://shop.test/p/coffee-maker-12345",
            bodyText='{"productUrl":"https://shop.test/p/coffee-maker-12345","image":"https://img.test/a.jpg","price":"CHF 99.90","description":"This is a long enough description for extraction."}',
        ),
        SavedSearchRenderedNetworkCapture(
            bodyText='{"items":[{"url":"https://shop.test/p/espresso-9999","mainImage":"https://img.test/b.jpg"}]}'
        ),
    ]
    extracted = extract_candidate_urls_from_render_captures(
        captures=captures,
        source_url="https://shop.test/list",
        max_items=10,
        normalize_url=lambda base, href: canonicalize_url(
            str(httpx.URL(base).join(href))
        ),
    )
    assert len(extracted) == 2
    first = next(item for item in extracted if "coffee-maker-12345" in item.url)
    assert first.price == "CHF 99.90"
    assert first.description is not None
    assert "https://img.test/a.jpg" in first.image_urls


def test_merge_precedence_and_image_max5():
    dom = [
        SavedSearchCandidate(
            url="https://shop.test/p/a-1234",
            title="DOM title",
            imageUrls=["https://img.test/1.jpg", "https://img.test/2.jpg"],
            price="CHF 10",
            description="dom desc",
        )
    ]
    network = [
        SavedSearchCandidate(
            url="https://shop.test/p/a-1234",
            title="Network title",
            imageUrls=["https://img.test/2.jpg", "https://img.test/3.jpg"],
            price="CHF 20",
            description="network desc",
        )
    ]
    product_list = [
        SavedSearchCandidate(
            url="https://shop.test/p/a-1234",
            title="ProductList title",
            imageUrls=[
                "https://img.test/4.jpg",
                "https://img.test/5.jpg",
                "https://img.test/6.jpg",
            ],
            price="CHF 30",
            description="productlist desc",
        )
    ]
    merged = merge_candidates_with_precedence(dom, network, product_list)
    assert len(merged) == 1
    candidate = merged[0]
    assert candidate.title == "ProductList title"
    assert candidate.price == "CHF 30"
    assert candidate.description == "productlist desc"
    assert candidate.image_urls == [
        "https://img.test/1.jpg",
        "https://img.test/2.jpg",
        "https://img.test/3.jpg",
        "https://img.test/4.jpg",
        "https://img.test/5.jpg",
    ]


@pytest.mark.asyncio
async def test_template_params_apply_q_and_filter_merge_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        html = '<a href="/p/product-1234">Product 1234</a>'
        return httpx.Response(200, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="template-source",
            urls=[
                {
                    "rawUrl": "https://shop.test/list?filter=100=1",
                    "templateParams": {
                        "q": "{search_term}",
                        "filter": "200=2|3",
                    },
                }
            ],
        )
        result = await service.ingest_source(
            source=source,
            search_term="mini fridge",
            max_items=10,
        )

    parsed_query = parse_qs(urlparse(result.source_urls[0]).query)
    assert parsed_query.get("q") == ["mini fridge"]
    assert parsed_query.get("filter") == ["100=1,200=2|3"]


@pytest.mark.asyncio
async def test_legacy_search_param_key_is_still_supported():
    def handler(request: httpx.Request) -> httpx.Response:
        html = '<a href="/p/product-1234">Product 1234</a>'
        return httpx.Response(200, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="legacy-source",
            urls=[
                {
                    "rawUrl": "https://shop.test/list",
                    "searchParamKey": "q",
                }
            ],
        )
        result = await service.ingest_source(
            source=source,
            search_term="espresso machine",
            max_items=10,
        )

    parsed_query = parse_qs(urlparse(result.source_urls[0]).query)
    assert parsed_query.get("q") == ["espresso machine"]


@pytest.mark.asyncio
async def test_duplicate_filter_entry_is_ignored_when_equivalent_to_template_param():
    def handler(request: httpx.Request) -> httpx.Response:
        html = '<a href="/p/product-1234">Product 1234</a>'
        return httpx.Response(200, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="dedupe-source",
            urls=[
                {
                    "rawUrl": "https://shop.test/list?filter=100=1",
                    "templateParams": {
                        "q": "{search_term}",
                        "filter": "200=2|3",
                    },
                }
            ],
            searchFilterConfig={
                "entries": [
                    {
                        "domain": "generic",
                        "value": "200=3|2",
                        "enabled": True,
                    }
                ]
            },
        )
        result = await service.ingest_source(
            source=source,
            search_term="fridge",
            max_items=10,
        )

    parsed_query = parse_qs(urlparse(result.source_urls[0]).query)
    assert parsed_query.get("q") == ["fridge"]
    assert parsed_query.get("filter") == ["100=1,200=2|3"]


@pytest.mark.asyncio
async def test_non_filter_entry_value_is_appended_to_search_query():
    def handler(request: httpx.Request) -> httpx.Response:
        html = '<a href="/p/product-1234">Product 1234</a>'
        return httpx.Response(200, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="search-suffix-source",
            urls=[
                {
                    "rawUrl": "https://www.fust.ch/search?q={search_term}",
                    "templateParams": {"q": "{search_term}"},
                }
            ],
            searchFilterConfig={
                "entries": [
                    {
                        "domain": "fust",
                        "value": ":relevance:Energieeffizienzklasse:A",
                        "enabled": True,
                    }
                ]
            },
        )
        result = await service.ingest_source(
            source=source,
            search_term="Kühlschrank",
            max_items=10,
        )

    parsed_query = parse_qs(urlparse(result.source_urls[0]).query)
    assert parsed_query.get("q") == ["Kühlschrank:relevance:Energieeffizienzklasse:A"]


@pytest.mark.asyncio
async def test_template_filter_encoded_value_is_not_double_encoded():
    def handler(request: httpx.Request) -> httpx.Response:
        html = '<a href="/p/product-1234">Product 1234</a>'
        return httpx.Response(200, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="encoded-filter-source",
            urls=[
                {
                    "rawUrl": "https://www.galaxus.ch/de/search",
                    "templateParams": {
                        "q": "{search_term}",
                        "filter": "21826%3D892",
                    },
                }
            ],
        )
        result = await service.ingest_source(
            source=source,
            search_term="Kühlschrank",
            max_items=5,
        )

    resolved_url = result.source_urls[0]
    assert "filter=21826%3D892" in resolved_url
    assert "filter=21826%253D892" not in resolved_url


@pytest.mark.asyncio
async def test_extraction_does_not_require_hardcoded_product_path_when_include_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        html = '<a href="/shop/fridge-alpha">Fridge Alpha Offer</a>'
        return httpx.Response(200, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="path-agnostic-source",
            urls=[{"rawUrl": "https://shop.test/list"}],
            searchFilterConfig={
                "candidateUrlIncludeSubstrings": [],
                "candidateUrlExcludeSubstrings": [],
            },
        )
        result = await service.ingest_source(
            source=source,
            search_term="fridge",
            max_items=10,
        )

    assert [candidate.url for candidate in result.candidates] == [
        "https://shop.test/shop/fridge-alpha"
    ]


@pytest.mark.asyncio
async def test_include_filters_remain_authoritative_after_path_gate_removal():
    def handler(request: httpx.Request) -> httpx.Response:
        html = '<a href="/shop/fridge-alpha">Fridge Alpha Offer</a>'
        return httpx.Response(200, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="include-authority-source",
            urls=[{"rawUrl": "https://shop.test/list"}],
            searchFilterConfig={
                "candidateUrlIncludeSubstrings": ["/p/"],
                "candidateUrlExcludeSubstrings": [],
            },
        )
        result = await service.ingest_source(
            source=source,
            search_term="fridge",
            max_items=10,
        )

    assert result.candidates == []


@pytest.mark.asyncio
async def test_failed_http_fetch_is_not_counted_as_fetched():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="http-failure-source",
            urls=[{"rawUrl": "https://shop.test/list"}],
        )
        result = await service.ingest_source(
            source=source,
            search_term="fridge",
            max_items=5,
        )

    assert len(result.url_diagnostics) == 1
    assert result.url_diagnostics[0].fetched == 0


@pytest.mark.asyncio
async def test_ingest_continues_after_one_template_transport_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        if "fail=1" in str(request.url):
            raise httpx.TransportError("simulated transport error")
        html = '<a href="/p/product-1234">Product 1234</a>'
        return httpx.Response(200, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="partial-failure-source",
            urls=[
                {"rawUrl": "https://shop.test/list?fail=1"},
                {"rawUrl": "https://shop.test/list"},
            ],
        )
        result = await service.ingest_source(
            source=source,
            search_term="fridge",
            max_items=10,
        )

    assert [candidate.url for candidate in result.candidates] == [
        "https://shop.test/p/product-1234"
    ]
    assert len(result.url_diagnostics) == 2
    assert result.url_diagnostics[0].error is not None
    assert result.url_diagnostics[0].parsed == 0
    assert result.url_diagnostics[1].error is None
