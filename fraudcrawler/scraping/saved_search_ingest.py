from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

import httpx

from fraudcrawler.scraping.saved_search_models import WebsiteSource
from fraudcrawler.scraping.search import (
    SavedSearchIngestResult,
    WebsiteSearch,
)
from fraudcrawler.scraping.zyte import SavedSearchRenderedPageResult

logger = logging.getLogger(__name__)


class SavedSearchIngestService:
    """Deprecated compatibility wrapper for website-source ingestion.

    Prefer using `WebsiteSearch.ingest_source(...)` directly.
    """

    def __init__(self, http_client: httpx.AsyncClient):
        self._http_client = http_client

    async def ingest_source(
        self,
        source: WebsiteSource,
        search_term: Optional[str],
        max_items: int,
        render_fetcher: Optional[
            Callable[..., Awaitable[SavedSearchRenderedPageResult]]
        ] = None,
    ) -> SavedSearchIngestResult:
        if render_fetcher is None:
            raise ValueError("SavedSearchIngestService requires a render_fetcher.")

        website_search = WebsiteSearch(
            http_client=self._http_client,
            zyteapi_key="deprecated-compat-layer",
            redis_use_cache=False,
        )
        setattr(website_search._zyteapi, "fetch_rendered_page", render_fetcher)
        result = await website_search.ingest_source(
            source=source,
            search_term=search_term,
            max_items=max_items,
        )

        # Keep parity with previous behavior: surface source-level failure quickly.
        if not result.candidates and result.url_diagnostics:
            if all(diag.render_error for diag in result.url_diagnostics):
                logger.debug(
                    "SavedSearchIngestService compatibility run returned only errors: %s",
                    [diag.model_dump() for diag in result.url_diagnostics],
                )
        return result
