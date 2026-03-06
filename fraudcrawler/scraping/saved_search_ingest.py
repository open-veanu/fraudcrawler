from __future__ import annotations

import logging
from time import perf_counter
from typing import Awaitable, Callable, List, Optional, Sequence, Tuple
from urllib.parse import (
    ParseResult,
    parse_qsl,
    quote,
    urlencode,
    urlparse,
    urlunparse,
)

import httpx

from fraudcrawler.scraping.saved_search_extraction import (
    extract_candidate_offers,
    extract_candidate_urls_from_render_captures,
    extract_candidates_from_rendered_product_list,
)
from fraudcrawler.scraping.url import filter_tracking_query_entries
from fraudcrawler.scraping.saved_search_filters import (
    apply_candidate_url_pattern_filters,
)
from fraudcrawler.scraping.saved_search_models import (
    SavedSearchCandidate,
    SavedSearchIngestResult,
    SavedSearchRenderedPageResult,
    SavedSearchSearchableUrl,
    SavedSearchRenderTriggerPolicy,
    SavedSearchSource,
    SavedSearchUrlTemplate,
    SavedSearchUrlDiagnostic,
)

logger = logging.getLogger(__name__)

SAVED_SEARCH_QUERY_PARAM_KEYS = ["q", "query", "keyword", "search"]
MAX_IMAGE_URLS_PER_CANDIDATE = 5


def get_search_param_order(param_key: str) -> int:
    try:
        return SAVED_SEARCH_QUERY_PARAM_KEYS.index(param_key.lower())
    except ValueError:
        return 10_000


def canonicalize_url(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip())
    host = parsed.hostname.lower() if parsed.hostname else ""
    netloc = host
    if parsed.port and not (
        (parsed.scheme == "https" and parsed.port == 443)
        or (parsed.scheme == "http" and parsed.port == 80)
    ):
        netloc = f"{host}:{parsed.port}"

    query_entries = [entry for entry in parse_qsl(parsed.query, keep_blank_values=True)]
    query_entries = filter_tracking_query_entries(query_entries)
    query_entries = sorted(
        query_entries,
        key=lambda item: (get_search_param_order(item[0]), item[0].lower(), item[1]),
    )
    query = urlencode(query_entries, quote_via=quote)
    cleaned = ParseResult(
        scheme=parsed.scheme,
        netloc=netloc,
        path=parsed.path,
        params=parsed.params,
        query=query,
        fragment="",
    )
    return urlunparse(cleaned)


def normalize_url(base_url: str, href: str) -> Optional[str]:
    try:
        base = httpx.URL(base_url)
        joined = base.join(href)
    except Exception:
        return None
    try:
        return canonicalize_url(str(joined))
    except Exception:
        return None


def _interpolate_search_term(value: str, search_term: Optional[str]) -> str:
    if "{search_term}" not in value:
        return value
    encoded = quote(search_term or "", safe="")
    return value.replace("{search_term}", encoded)


def build_searchable_url(
    template: SavedSearchUrlTemplate,
    searchable_url: SavedSearchSearchableUrl,
    search_term: Optional[str],
) -> Optional[str]:
    interpolated = _interpolate_search_term(searchable_url.filter_url, search_term)
    return normalize_url(base_url=template.base_url, href=interpolated)


def merge_candidate_image_urls(*lists: Sequence[str]) -> List[str]:
    merged = []
    seen = set()
    for lst in lists:
        for raw in lst:
            txt = str(raw or "").strip()
            if not txt or txt in seen:
                continue
            if not txt.startswith(("http://", "https://")):
                continue
            seen.add(txt)
            merged.append(txt)
            if len(merged) >= MAX_IMAGE_URLS_PER_CANDIDATE:
                return merged
    return merged


def merge_text(
    existing_value: Optional[str], incoming_value: Optional[str], prefer_incoming: bool
) -> Optional[str]:
    existing = (existing_value or "").strip()
    incoming = (incoming_value or "").strip()
    if prefer_incoming:
        return incoming or existing or None
    return existing or incoming or None


def merge_candidates_with_precedence(
    dom_candidates: Sequence[SavedSearchCandidate],
    network_candidates: Sequence[SavedSearchCandidate],
    product_list_candidates: Sequence[SavedSearchCandidate],
) -> List[SavedSearchCandidate]:
    merged = {}
    order = []

    def upsert(
        candidate: SavedSearchCandidate,
        prefer_text: bool,
        title_preferred: bool = False,
    ) -> None:
        key = candidate.url
        if key not in merged:
            merged[key] = SavedSearchCandidate(
                url=candidate.url,
                title=candidate.title,
                imageUrls=merge_candidate_image_urls(candidate.image_urls),
                price=candidate.price,
                description=candidate.description,
            )
            order.append(key)
            return
        existing = merged[key]
        merged[key] = SavedSearchCandidate(
            url=existing.url,
            title=candidate.title
            if title_preferred and candidate.title
            else existing.title,
            imageUrls=merge_candidate_image_urls(
                existing.image_urls, candidate.image_urls
            ),
            price=merge_text(
                existing.price, candidate.price, prefer_incoming=prefer_text
            ),
            description=merge_text(
                existing.description, candidate.description, prefer_incoming=prefer_text
            ),
        )

    for cand in dom_candidates:
        upsert(cand, prefer_text=False)
    for cand in network_candidates:
        upsert(cand, prefer_text=True)
    for cand in product_list_candidates:
        upsert(cand, prefer_text=True, title_preferred=True)
    return [merged[key] for key in order]


def should_trigger_render_fallback(
    trigger_policy: SavedSearchRenderTriggerPolicy,
    static_http_status: Optional[int],
    post_filter_count: int,
) -> bool:
    if trigger_policy == "always":
        return True
    if trigger_policy == "on_http_403":
        return static_http_status == 403
    return post_filter_count == 0


class SavedSearchIngestService:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
    ):
        self._http_client = http_client

    async def _fetch_listing_page(self, url: str) -> Tuple[int, str]:
        response = await self._http_client.get(url)
        if not response.is_success:
            return response.status_code, ""
        return response.status_code, response.text

    async def ingest_source(
        self,
        source: SavedSearchSource,
        search_term: Optional[str],
        max_items: int,
        render_fetcher: Optional[
            Callable[..., Awaitable[SavedSearchRenderedPageResult]]
        ] = None,
    ) -> SavedSearchIngestResult:
        combined: List[SavedSearchCandidate] = []
        seen: dict[str, int] = {}
        diagnostics: List[SavedSearchUrlDiagnostic] = []
        resolved_urls: List[str] = []

        for template in source.urls:
            for searchable_url in template.searchable_urls:
                resolved: Optional[str] = None
                try:
                    resolved = build_searchable_url(
                        template=template,
                        searchable_url=searchable_url,
                        search_term=search_term,
                    )
                    if resolved is None:
                        raise ValueError(
                            f"Failed to resolve searchableUrl='{searchable_url.filter_url}' "
                            f"against baseUrl='{template.base_url}'."
                        )
                    resolved_urls.append(resolved)
                    logger.info(
                        "Saved-search ingest URL resolved | source=%s | filter_url=%s | resolved=%s",
                        source.name,
                        searchable_url.filter_url,
                        resolved,
                    )

                    static_http_status: Optional[int] = None
                    static_http_status, html = await self._fetch_listing_page(resolved)
                    extraction = extract_candidate_offers(
                        html=html,
                        source_url=resolved,
                        max_items=max_items,
                        normalize_url=normalize_url,
                    )
                    logger.info(
                        "Static HTML extraction completed | source=%s | status=%s | candidates=%s",
                        source.name,
                        static_http_status,
                        extraction.candidates_after,
                    )

                    logger.info(
                        "Candidate URLs before pattern filtering | source=%s | urls=%s",
                        source.name,
                        [candidate.url for candidate in extraction.candidates],
                    )
                    filtered = apply_candidate_url_pattern_filters(
                        extraction.candidates,
                        include_substrings=searchable_url.include_substrings,
                        exclude_substrings=searchable_url.exclude_substrings,
                    )
                    final_candidates = filtered.filtered_candidates
                    merged_candidate_count = len(final_candidates)
                    product_list_count = 0
                    product_list_mapped_count = 0

                    render_attempted = False
                    render_used = False
                    render_error = None
                    render_provider = None
                    render_trigger = None
                    render_http_status = None

                    render_config = (
                        source.search_filter_config.render_fallback
                        if source.search_filter_config
                        else None
                    )
                    if render_config and render_config.enabled:
                        render_trigger = render_config.trigger_policy
                        should_try = should_trigger_render_fallback(
                            trigger_policy=render_config.trigger_policy,
                            static_http_status=static_http_status,
                            post_filter_count=len(final_candidates),
                        )
                        logger.info(
                            (
                                "Render fallback decision | source=%s | should_try=%s | "
                                "policy=%s | post_filter_count=%s | static_http_status=%s"
                            ),
                            source.name,
                            should_try,
                            render_config.trigger_policy,
                            len(final_candidates),
                            static_http_status,
                        )
                        if should_try and render_fetcher:
                            render_attempted = True
                            render_provider = render_config.provider
                            try:
                                started = perf_counter()
                                rendered = await render_fetcher(
                                    url=resolved,
                                    provider=render_config.provider,
                                    javascript=render_config.javascript,
                                    include_iframes=render_config.include_iframes,
                                    actions=render_config.actions or [],
                                    network_capture=render_config.network_capture or [],
                                    request_headers=(
                                        render_config.request_headers.model_dump(
                                            exclude_none=True
                                        )
                                        if render_config.request_headers
                                        else None
                                    ),
                                )
                                _ = perf_counter() - started
                                render_http_status = rendered.status_code
                                dom_candidates = extract_candidate_offers(
                                    html=rendered.html,
                                    source_url=resolved,
                                    max_items=max_items,
                                    normalize_url=normalize_url,
                                ).candidates
                                network_candidates = (
                                    extract_candidate_urls_from_render_captures(
                                        captures=rendered.network_captures,
                                        source_url=resolved,
                                        max_items=max_items,
                                        normalize_url=normalize_url,
                                    )
                                )
                                product_candidates = (
                                    extract_candidates_from_rendered_product_list(
                                        items=rendered.product_list_items,
                                        source_url=resolved,
                                        max_items=max_items,
                                        normalize_url=normalize_url,
                                    )
                                )
                                product_list_count = len(rendered.product_list_items)
                                product_list_mapped_count = len(product_candidates)
                                merged = merge_candidates_with_precedence(
                                    dom_candidates=dom_candidates,
                                    network_candidates=network_candidates,
                                    product_list_candidates=product_candidates,
                                )
                                merged_candidate_count = len(merged)
                                logger.info(
                                    "Rendered candidate URLs before pattern filtering | source=%s | urls=%s",
                                    source.name,
                                    [candidate.url for candidate in merged],
                                )
                                rendered_filtered = apply_candidate_url_pattern_filters(
                                    merged,
                                    include_substrings=searchable_url.include_substrings,
                                    exclude_substrings=searchable_url.exclude_substrings,
                                )
                                logger.info(
                                    (
                                        "Render fallback extracted | source=%s | provider=%s | "
                                        "dom=%s | network=%s | product_list=%s | product_list_mapped=%s | "
                                        "merged=%s | filtered=%s"
                                    ),
                                    source.name,
                                    render_provider,
                                    len(dom_candidates),
                                    len(network_candidates),
                                    product_list_count,
                                    product_list_mapped_count,
                                    merged_candidate_count,
                                    len(rendered_filtered.filtered_candidates),
                                )
                                if rendered_filtered.filtered_candidates:
                                    final_candidates = (
                                        rendered_filtered.filtered_candidates
                                    )
                                    render_used = True
                            except Exception as err:
                                render_error = str(err)
                                logger.warning(
                                    "Render fallback failed | source=%s | provider=%s | error=%s",
                                    source.name,
                                    render_provider,
                                    render_error,
                                )

                    deduped_count = 0
                    for candidate in final_candidates:
                        if candidate.url in seen:
                            idx = seen[candidate.url]
                            existing = combined[idx]
                            combined[idx] = SavedSearchCandidate(
                                url=existing.url,
                                title=existing.title,
                                imageUrls=merge_candidate_image_urls(
                                    existing.image_urls, candidate.image_urls
                                ),
                                price=merge_text(existing.price, candidate.price, True),
                                description=merge_text(
                                    existing.description, candidate.description, True
                                ),
                            )
                            deduped_count += 1
                            continue
                        seen[candidate.url] = len(combined)
                        combined.append(candidate)

                    fetched = (
                        1
                        if static_http_status is not None
                        and 200 <= static_http_status < 300
                        else 0
                    )
                    diagnostics.append(
                        SavedSearchUrlDiagnostic(
                            url=searchable_url.filter_url,
                            resolvedUrl=resolved,
                            extractionStrategy=extraction.strategy,
                            candidatesBefore=extraction.candidates_before,
                            candidatesAfter=extraction.candidates_after,
                            candidateUrlPatternIncludeCount=filtered.include_count,
                            candidateUrlPatternExcludeCount=filtered.exclude_count,
                            droppedByMissingIncludeAllMatch=filtered.dropped_by_missing_include_all_match,
                            droppedByExcludeSubstring=filtered.dropped_by_exclude_substring,
                            firstDroppedByMissingIncludeSubstring=filtered.first_dropped_by_missing_include_substring,
                            firstDroppedByExcludeSubstring=filtered.first_dropped_by_exclude_substring,
                            renderFallbackAttempted=render_attempted,
                            renderFallbackUsed=render_used,
                            renderFallbackProvider=render_provider,
                            renderFallbackTriggerPolicy=render_trigger,
                            renderFallbackHttpStatus=render_http_status,
                            renderFallbackError=render_error,
                            productListCount=product_list_count,
                            productListMappedCount=product_list_mapped_count,
                            mergedCandidateCount=merged_candidate_count,
                            fetched=fetched,
                            parsed=len(final_candidates),
                            deduped=deduped_count,
                        )
                    )
                    logger.info(
                        (
                            "Saved-search URL finished | source=%s | final_candidates=%s | "
                            "fallback_attempted=%s | fallback_used=%s | fallback_error=%s"
                        ),
                        source.name,
                        len(final_candidates),
                        render_attempted,
                        render_used,
                        render_error,
                    )
                except Exception as err:
                    logger.warning(
                        (
                            "Saved-search ingest failed for searchable URL | source=%s | "
                            "base_url=%s | filter_url=%s | resolved=%s | error=%s"
                        ),
                        source.name,
                        template.base_url,
                        searchable_url.filter_url,
                        resolved,
                        err,
                        exc_info=True,
                    )
                    diagnostics.append(
                        SavedSearchUrlDiagnostic(
                            url=searchable_url.filter_url,
                            resolvedUrl=resolved,
                            fetched=0,
                            parsed=0,
                            deduped=0,
                            error=str(err),
                        )
                    )

        return SavedSearchIngestResult(
            sourceName=source.name,
            sourceUrls=resolved_urls,
            candidates=combined,
            samples=combined[:10],
            fetched=sum(item.fetched for item in diagnostics),
            parsed=len(combined),
            deduped=sum(item.deduped for item in diagnostics),
            urlDiagnostics=diagnostics,
        )
