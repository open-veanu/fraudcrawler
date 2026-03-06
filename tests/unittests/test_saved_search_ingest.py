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
            urls=[
                {
                    "baseUrl": "https://shop.test/",
                    "searchableUrls": [{"filterUrl": "list?q={search_term}"}],
                }
            ],
            useGraph=True,
            graphURL="https://shop.test/graphql",
        )


def test_old_graphql_filter_config_field_is_rejected():
    with pytest.raises(ValidationError):
        SavedSearchSource(
            name="invalid-source",
            urls=[
                {
                    "baseUrl": "https://shop.test/",
                    "searchableUrls": [{"filterUrl": "list?q={search_term}"}],
                }
            ],
            searchFilterConfig={"graphql": {"enabled": True}},
        )


def test_old_url_template_fields_are_rejected():
    with pytest.raises(ValidationError):
        SavedSearchSource(
            name="invalid-template",
            urls=[
                {
                    "rawUrl": "https://shop.test/list",
                    "templateParams": {"q": "{search_term}"},
                }
            ],
        )


def test_include_exclude_filters_include_all_and_exclude_any():
    candidates = [
        SavedSearchCandidate(url="https://shop.test/s1/product/a-123", title="A"),
        SavedSearchCandidate(url="https://shop.test/s1/category/a-123", title="B"),
        SavedSearchCandidate(url="https://shop.test/s1/product/blocked-123", title="C"),
    ]
    source = SavedSearchSource(
        name="test",
        urls=[
            {
                "baseUrl": "https://shop.test/",
                "searchableUrls": [
                    {
                        "filterUrl": "list",
                        "includeSubstrings": ["/s1/product/"],
                        "excludeSubstrings": ["BLOCKED"],
                    }
                ],
            }
        ],
    )
    searchable = source.urls[0].searchable_urls[0]
    result = apply_candidate_url_pattern_filters(
        candidates,
        include_substrings=searchable.include_substrings,
        exclude_substrings=searchable.exclude_substrings,
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


def test_normalize_url_preserves_trailing_slash_for_index_paths():
    normalized = normalize_url(
        base_url="https://www.frankenspalter.ch/",
        href=(
            "de/catalogsearch/result/index/"
            "?q=k%C3%BChlschrank&itg_202303151629545861692577=1818"
        ),
    )
    assert (
        normalized
        == "https://www.frankenspalter.ch/de/catalogsearch/result/index/?q=k%C3%BChlschrank&itg_202303151629545861692577=1818"
    )


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
async def test_searchable_url_applies_search_term_and_preserves_query_params():
    def handler(request: httpx.Request) -> httpx.Response:
        html = '<a href="/p/product-1234">Product 1234</a>'
        return httpx.Response(200, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        service = SavedSearchIngestService(http_client=client)
        source = SavedSearchSource(
            name="frankenspalter-like-source",
            urls=[
                {
                    "baseUrl": "https://www.frankenspalter.ch/",
                    "searchableUrls": [
                        {
                            "filterUrl": "de/catalogsearch/result/index/?q={search_term}&itg_202303151629545861692577=1818",
                            "includeSubstrings": [],
                            "excludeSubstrings": [],
                        }
                    ],
                }
            ],
        )
        result = await service.ingest_source(
            source=source,
            search_term="Kühlschrank",
            max_items=5,
        )

    parsed_query = parse_qs(urlparse(result.source_urls[0]).query)
    assert parsed_query.get("q") == ["Kühlschrank"]
    assert parsed_query.get("itg_202303151629545861692577") == ["1818"]
    assert "filter=" not in result.source_urls[0]


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
            urls=[
                {
                    "baseUrl": "https://shop.test/",
                    "searchableUrls": [
                        {
                            "filterUrl": "list",
                            "includeSubstrings": [],
                            "excludeSubstrings": [],
                        }
                    ],
                }
            ],
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
            urls=[
                {
                    "baseUrl": "https://shop.test/",
                    "searchableUrls": [
                        {
                            "filterUrl": "list",
                            "includeSubstrings": ["/p/"],
                            "excludeSubstrings": [],
                        }
                    ],
                }
            ],
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
            urls=[
                {
                    "baseUrl": "https://shop.test/",
                    "searchableUrls": [
                        {
                            "filterUrl": "list",
                            "includeSubstrings": [],
                            "excludeSubstrings": [],
                        }
                    ],
                }
            ],
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
                {
                    "baseUrl": "https://shop.test/",
                    "searchableUrls": [
                        {
                            "filterUrl": "list?fail=1",
                            "includeSubstrings": [],
                            "excludeSubstrings": [],
                        }
                    ],
                },
                {
                    "baseUrl": "https://shop.test/",
                    "searchableUrls": [
                        {
                            "filterUrl": "list",
                            "includeSubstrings": [],
                            "excludeSubstrings": [],
                        }
                    ],
                },
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
