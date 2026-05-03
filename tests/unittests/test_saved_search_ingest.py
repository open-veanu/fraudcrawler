import httpx
import pytest
from pydantic import ValidationError

from fraudcrawler.scraping.search import WebsiteSearch
from fraudcrawler.scraping.saved_search_models import (
    WebsiteSource,
)
from fraudcrawler.scraping.zyte import (
    SavedSearchRenderedPageResult,
    SavedSearchRenderedProductListItem,
)


def _source_with_render_options(
    *,
    filter_url: str = "list?q={search_term}",
    include_substrings: list[str] | None = None,
    exclude_substrings: list[str] | None = None,
) -> WebsiteSource:
    return WebsiteSource.model_validate(
        {
            "name": "Test Source",
            "urls": [
                {
                    "baseUrl": "https://shop.test/",
                    "searchableUrls": [
                        {
                            "filterUrl": filter_url,
                            "includeSubstrings": include_substrings or [],
                            "excludeSubstrings": exclude_substrings or [],
                        }
                    ],
                }
            ],
            "searchFilterConfig": {
                "renderOptions": {
                    "javascript": True,
                    "includeIframes": False,
                    "actions": [],
                    "networkCapture": [],
                }
            },
        }
    )


def _rendered_page(
    *,
    status_code: int = 200,
    product_list_items: list[SavedSearchRenderedProductListItem] | None = None,
) -> SavedSearchRenderedPageResult:
    return SavedSearchRenderedPageResult(
        html="<html></html>",
        statusCode=status_code,
        elapsedMs=0,
        actionStatuses=[],
        actionError=None,
        productListItems=product_list_items or [],
    )


def test_old_render_fallback_config_is_rejected():
    with pytest.raises(ValidationError):
        WebsiteSource.model_validate(
            {
                "name": "invalid-render-fallback",
                "urls": [
                    {
                        "baseUrl": "https://shop.test/",
                        "searchableUrls": [{"filterUrl": "list?q={search_term}"}],
                    }
                ],
                "searchFilterConfig": {
                    "renderFallback": {
                        "enabled": True,
                        "provider": "zyte",
                        "triggerPolicy": "always",
                    }
                },
            }
        )


@pytest.mark.asyncio
async def test_ingest_source_maps_product_list_candidates():
    async with httpx.AsyncClient() as client:
        website_search = WebsiteSearch(
            http_client=client,
            zyteapi_key="test-key",
            redis_use_cache=False,
        )
        source = _source_with_render_options()

        async def render_fetcher(**kwargs):
            assert kwargs["url"] == "https://shop.test/list?q=mini+fridge"
            return _rendered_page(
                product_list_items=[
                    SavedSearchRenderedProductListItem(
                        url="/p/123",
                        name="Mini Fridge",
                        price="CHF 499",
                        images=["/img/a.jpg"],
                    )
                ]
            )

        website_search._zyteapi.fetch_rendered_page = render_fetcher  # type: ignore[method-assign,assignment]
        result = await website_search.ingest_source(
            source=source,
            search_term="mini fridge",
            max_items=5,
        )

    assert [candidate.url for candidate in result.candidates] == [
        "https://shop.test/p/123"
    ]
    assert result.candidates[0].title == "Mini Fridge"
    assert result.candidates[0].price == "CHF 499"
    assert result.url_diagnostics[0].render_http_status == 200
    assert result.url_diagnostics[0].render_error is None


@pytest.mark.asyncio
async def test_ingest_source_applies_include_and_exclude_url_filters():
    async with httpx.AsyncClient() as client:
        website_search = WebsiteSearch(
            http_client=client,
            zyteapi_key="test-key",
            redis_use_cache=False,
        )
        source = _source_with_render_options(
            include_substrings=["/p/"],
            exclude_substrings=["blocked"],
        )

        async def render_fetcher(**kwargs):
            return _rendered_page(
                product_list_items=[
                    SavedSearchRenderedProductListItem(url="/p/ok", name="ok"),
                    SavedSearchRenderedProductListItem(url="/x/nope", name="nope"),
                    SavedSearchRenderedProductListItem(
                        url="/p/blocked-1", name="blocked"
                    ),
                ]
            )

        website_search._zyteapi.fetch_rendered_page = render_fetcher  # type: ignore[method-assign,assignment]
        result = await website_search.ingest_source(
            source=source,
            search_term="k",
            max_items=10,
        )

    assert [candidate.url for candidate in result.candidates] == [
        "https://shop.test/p/ok"
    ]


@pytest.mark.asyncio
async def test_ingest_source_records_render_errors():
    async with httpx.AsyncClient() as client:
        website_search = WebsiteSearch(
            http_client=client,
            zyteapi_key="test-key",
            redis_use_cache=False,
        )
        source = _source_with_render_options()

        async def render_fetcher(**kwargs):
            raise RuntimeError("render failed")

        website_search._zyteapi.fetch_rendered_page = render_fetcher  # type: ignore[method-assign,assignment]
        result = await website_search.ingest_source(
            source=source,
            search_term="k",
            max_items=10,
        )

    assert result.candidates == []
    assert len(result.url_diagnostics) == 1
    assert result.url_diagnostics[0].render_error == "render failed"
