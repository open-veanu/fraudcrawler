import asyncio

import pytest

from fraudcrawler.base.base import Deepness, Enrichment, Language, Location, ProductItem
from fraudcrawler.base.orchestrator import Orchestrator
from fraudcrawler.scraping.saved_search_models import (
    WebsiteSource,
    WebsiteSourceUrlTemplate,
)
from fraudcrawler.scraping.search import (
    WebsiteSearch,
    SearchEngineName,
)


def test_build_website_source_engine_name_slugifies_source_name():
    assert (
        WebsiteSearch._build_website_source_engine_name("Boost Galaxus")
        == "boost_galaxus_search_engine"
    )


def test_build_website_source_engine_name_strips_non_ascii_and_symbols():
    assert (
        WebsiteSearch._build_website_source_engine_name("  BÖÖS Galaxus!!  ")
        == "boos_galaxus_search_engine"
    )


def test_build_website_source_engine_name_falls_back_when_empty():
    assert (
        WebsiteSearch._build_website_source_engine_name("!!!")
        == "website_source_search_engine"
    )


def test_search_engine_name_rejects_legacy_saved_search_token():
    with pytest.raises(ValueError):
        SearchEngineName("saved_search")


class _StopAfterSizing(Exception):
    """Signal exception to stop Orchestrator.run after worker sizing."""


class _DummyOrchestrator(Orchestrator):
    async def _collect_results(
        self,
        queue_in: asyncio.Queue[dict | None],  # type: ignore[override]
    ) -> None:
        return None


def _saved_source(name: str) -> WebsiteSource:
    return WebsiteSource(
        name=name,
        urls=[
            WebsiteSourceUrlTemplate.model_validate(
                {
                    "baseUrl": "https://example.com/",
                    "searchableUrls": [
                        {
                            "filterUrl": "search?q={search_term}",
                            "includeSubstrings": [],
                            "excludeSubstrings": [],
                        }
                    ],
                }
            )
        ],
    )


@pytest.mark.asyncio
async def test_orchestrator_sizing_includes_website_source_sources(monkeypatch):
    orchestrator = _DummyOrchestrator(
        searcher=object(),  # type: ignore[arg-type]
        enricher=object(),  # type: ignore[arg-type]
        url_collector=object(),  # type: ignore[arg-type]
        zyteapi=object(),  # type: ignore[arg-type]
        processor=object(),  # type: ignore[arg-type]
        n_srch_wkrs=4,
        n_cntx_wkrs=2,
        n_proc_wkrs=2,
    )
    captured: dict[str, int] = {}

    def _capture_setup(
        n_srch_wkrs: int,
        n_cntx_wkrs: int,
        n_proc_wkrs: int,
    ) -> None:
        captured["n_srch_wkrs"] = n_srch_wkrs
        captured["n_cntx_wkrs"] = n_cntx_wkrs
        captured["n_proc_wkrs"] = n_proc_wkrs
        raise _StopAfterSizing()

    monkeypatch.setattr(orchestrator, "_setup_async_framework", _capture_setup)

    with pytest.raises(_StopAfterSizing):
        await orchestrator.run(
            search_term="kuehlschrank",
            search_engines=[
                SearchEngineName.GOOGLE,
                SearchEngineName.TOPPREISE,
                SearchEngineName.WEBSITE_SOURCE,
            ],
            language=Language(name="German"),
            location=Location(name="Switzerland"),
            deepness=Deepness(
                num_results=10,
                enrichment=Enrichment(additional_terms=1, additional_urls_per_term=10),
            ),
            website_source_sources=[_saved_source(f"source-{idx}") for idx in range(5)],
        )

    # estimated items: engines * (initial + enriched_terms) + saved_sources
    #                = 2 * (1 + 1) + 5 = 9, capped at configured n_srch_wkrs=4
    assert captured["n_srch_wkrs"] == 4
    assert captured["n_cntx_wkrs"] == 2
    assert captured["n_proc_wkrs"] == 2


@pytest.mark.asyncio
async def test_orchestrator_sizing_baseline_without_saved_sources(monkeypatch):
    orchestrator = _DummyOrchestrator(
        searcher=object(),  # type: ignore[arg-type]
        enricher=object(),  # type: ignore[arg-type]
        url_collector=object(),  # type: ignore[arg-type]
        zyteapi=object(),  # type: ignore[arg-type]
        processor=object(),  # type: ignore[arg-type]
        n_srch_wkrs=8,
        n_cntx_wkrs=3,
        n_proc_wkrs=3,
    )
    captured: dict[str, int] = {}

    def _capture_setup(
        n_srch_wkrs: int,
        n_cntx_wkrs: int,
        n_proc_wkrs: int,
    ) -> None:
        captured["n_srch_wkrs"] = n_srch_wkrs
        captured["n_cntx_wkrs"] = n_cntx_wkrs
        captured["n_proc_wkrs"] = n_proc_wkrs
        raise _StopAfterSizing()

    monkeypatch.setattr(orchestrator, "_setup_async_framework", _capture_setup)

    with pytest.raises(_StopAfterSizing):
        await orchestrator.run(
            search_term="kuehlschrank",
            search_engines=[SearchEngineName.GOOGLE],
            language=Language(name="German"),
            location=Location(name="Switzerland"),
            deepness=Deepness(num_results=10),
            website_source_sources=None,
        )

    # estimated items: 1 engine * (1 initial + 0 enriched) + 0 sources = 1
    assert captured["n_srch_wkrs"] == 1
    assert captured["n_cntx_wkrs"] == 3
    assert captured["n_proc_wkrs"] == 3


@pytest.mark.asyncio
async def test_add_srch_items_handles_saved_search_in_engine_loop():
    orchestrator = _DummyOrchestrator(
        searcher=object(),  # type: ignore[arg-type]
        enricher=object(),  # type: ignore[arg-type]
        url_collector=object(),  # type: ignore[arg-type]
        zyteapi=object(),  # type: ignore[arg-type]
        processor=object(),  # type: ignore[arg-type]
    )
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    sources = [_saved_source("source-a"), _saved_source("source-b")]
    await orchestrator._add_srch_items(
        queue=queue,
        search_term="kuehlschrank",
        search_engines=[SearchEngineName.GOOGLE, SearchEngineName.WEBSITE_SOURCE],
        website_source_sources=sources,
        language=Language(name="German"),
        location=Location(name="Switzerland"),
        deepness=Deepness(num_results=10),
        marketplaces=None,
        excluded_urls=None,
    )

    items: list[dict] = []
    while not queue.empty():
        value = queue.get_nowait()
        if value is not None:
            items.append(value)

    google_items = [
        item for item in items if item["search_engine"] == SearchEngineName.GOOGLE
    ]
    saved_items = [
        item
        for item in items
        if item["search_engine"] == SearchEngineName.WEBSITE_SOURCE
    ]

    assert len(google_items) == 1
    assert len(saved_items) == 2
    assert {item["website_source_source"].name for item in saved_items} == {
        "source-a",
        "source-b",
    }


@pytest.mark.asyncio
async def test_add_srch_items_saved_search_without_sources_is_graceful():
    orchestrator = _DummyOrchestrator(
        searcher=object(),  # type: ignore[arg-type]
        enricher=object(),  # type: ignore[arg-type]
        url_collector=object(),  # type: ignore[arg-type]
        zyteapi=object(),  # type: ignore[arg-type]
        processor=object(),  # type: ignore[arg-type]
    )
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    await orchestrator._add_srch_items(
        queue=queue,
        search_term="kuehlschrank",
        search_engines=[SearchEngineName.WEBSITE_SOURCE],
        website_source_sources=None,
        language=Language(name="German"),
        location=Location(name="Switzerland"),
        deepness=Deepness(num_results=10),
        marketplaces=None,
        excluded_urls=None,
    )

    assert queue.empty()


@pytest.mark.asyncio
async def test_run_raises_when_sources_given_without_saved_search_engine():
    orchestrator = _DummyOrchestrator(
        searcher=object(),  # type: ignore[arg-type]
        enricher=object(),  # type: ignore[arg-type]
        url_collector=object(),  # type: ignore[arg-type]
        zyteapi=object(),  # type: ignore[arg-type]
        processor=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match=(
            "website_source_sources provided but search_engines does not include "
            "SearchEngineName.WEBSITE_SOURCE"
        ),
    ):
        await orchestrator.run(
            search_term="kuehlschrank",
            search_engines=[SearchEngineName.GOOGLE],
            language=Language(name="German"),
            location=Location(name="Switzerland"),
            deepness=Deepness(num_results=10),
            website_source_sources=[_saved_source("source-a")],
        )


@pytest.mark.asyncio
async def test_run_rejects_legacy_saved_search_sources_keyword():
    orchestrator = _DummyOrchestrator(
        searcher=object(),  # type: ignore[arg-type]
        enricher=object(),  # type: ignore[arg-type]
        url_collector=object(),  # type: ignore[arg-type]
        zyteapi=object(),  # type: ignore[arg-type]
        processor=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="saved_search_sources"):
        await orchestrator.run(  # type: ignore[call-arg]
            search_term="kuehlschrank",
            search_engines=[SearchEngineName.WEBSITE_SOURCE],
            language=Language(name="German"),
            location=Location(name="Switzerland"),
            deepness=Deepness(num_results=10),
            saved_search_sources=[_saved_source("source-a")],
        )


def test_check_exact_search_sets_match_for_quoted_terms():
    orchestrator = _DummyOrchestrator(
        searcher=object(),  # type: ignore[arg-type]
        enricher=object(),  # type: ignore[arg-type]
        url_collector=object(),  # type: ignore[arg-type]
        zyteapi=object(),  # type: ignore[arg-type]
        processor=object(),  # type: ignore[arg-type]
    )
    product = ProductItem(
        search_term='"mini fridge"',
        search_term_type="initial",
        url="https://shop.test/p/123",
        url_resolved="https://shop.test/p/123",
        search_engine_name="website_source",
        domain="shop.test",
        product_name="Mini Fridge 120L",
    )

    checked = orchestrator._check_exact_search(product=product)

    assert checked.exact_search is True
    assert checked.exact_search_match is True
