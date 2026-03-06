from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


SavedSearchRenderTriggerPolicy = Literal["on_zero_candidates", "always", "on_http_403"]
SavedSearchRenderProvider = Literal["generic", "zyte"]
SavedSearchExtractionStrategy = Literal["adapter", "generic"]


class SavedSearchRenderFallbackRequestHeaders(BaseModel):
    referer: Optional[str] = None


class SavedSearchRenderFallbackConfig(BaseModel):
    enabled: bool = True
    provider: SavedSearchRenderProvider = "generic"
    trigger_policy: SavedSearchRenderTriggerPolicy = Field(
        default="on_zero_candidates", alias="triggerPolicy"
    )
    javascript: Optional[bool] = None
    include_iframes: Optional[bool] = Field(default=None, alias="includeIframes")
    request_headers: Optional[SavedSearchRenderFallbackRequestHeaders] = Field(
        default=None, alias="requestHeaders"
    )
    actions: Optional[List[dict[str, Any]]] = None
    network_capture: Optional[List[dict[str, Any]]] = Field(
        default=None, alias="networkCapture"
    )


class SavedSearchSearchableUrl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter_url: str = Field(alias="filterUrl")
    include_substrings: List[str] = Field(
        default_factory=list, alias="includeSubstrings"
    )
    exclude_substrings: List[str] = Field(
        default_factory=list, alias="excludeSubstrings"
    )

    @field_validator("filter_url", mode="before")
    @classmethod
    def _normalize_filter_url(cls, value: Any) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("Saved-search filterUrl must not be empty.")
        return normalized

    @field_validator("include_substrings", "exclude_substrings", mode="before")
    @classmethod
    def _normalize_tokens(cls, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        deduped: List[str] = []
        seen = set()
        for token in value:
            txt = str(token).strip()
            if not txt:
                continue
            low = txt.lower()
            if low in seen:
                continue
            seen.add(low)
            deduped.append(txt)
        return deduped


class SavedSearchUrlTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(alias="baseUrl")
    searchable_urls: List[SavedSearchSearchableUrl] = Field(alias="searchableUrls")

    @field_validator("base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, value: Any) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("Saved-search baseUrl must not be empty.")
        return normalized


class SavedSearchFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    mode: str = "per_source_all_urls"
    render_fallback: Optional[SavedSearchRenderFallbackConfig] = Field(
        default=None, alias="renderFallback"
    )


class SavedSearchSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    urls: List[SavedSearchUrlTemplate]
    search_filter_config: Optional[SavedSearchFilterConfig] = Field(
        default=None, alias="searchFilterConfig"
    )


class SavedSearchCandidate(BaseModel):
    url: str
    title: str
    image_urls: List[str] = Field(default_factory=list, alias="imageUrls")
    price: Optional[str] = None
    description: Optional[str] = None


class SavedSearchExtractionResult(BaseModel):
    candidates: List[SavedSearchCandidate] = Field(default_factory=list)
    strategy: SavedSearchExtractionStrategy = "generic"
    adapter_name: Optional[str] = Field(default=None, alias="adapterName")
    candidates_before: int = Field(default=0, alias="candidatesBefore")
    candidates_after: int = Field(default=0, alias="candidatesAfter")
    top_reject_reason: Optional[str] = Field(default=None, alias="topRejectReason")


class SavedSearchPatternFilterResult(BaseModel):
    filtered_candidates: List[SavedSearchCandidate] = Field(
        default_factory=list, alias="filteredCandidates"
    )
    include_count: int = Field(default=0, alias="includeCount")
    exclude_count: int = Field(default=0, alias="excludeCount")
    dropped_by_missing_include_substring: int = Field(
        default=0, alias="droppedByMissingIncludeSubstring"
    )
    dropped_by_missing_include_all_match: int = Field(
        default=0, alias="droppedByMissingIncludeAllMatch"
    )
    dropped_by_exclude_substring: int = Field(
        default=0, alias="droppedByExcludeSubstring"
    )
    first_dropped_by_missing_include_substring: Optional[str] = Field(
        default=None, alias="firstDroppedByMissingIncludeSubstring"
    )
    first_dropped_by_exclude_substring: Optional[str] = Field(
        default=None, alias="firstDroppedByExcludeSubstring"
    )


class SavedSearchRenderedNetworkCapture(BaseModel):
    url: Optional[str] = None
    status_code: Optional[int] = Field(default=None, alias="statusCode")
    content_type: Optional[str] = Field(default=None, alias="contentType")
    body_text: Optional[str] = Field(default=None, alias="bodyText")


class SavedSearchRenderedProductListItem(BaseModel):
    url: Optional[str] = None
    name: Optional[str] = None
    price: Optional[str | float | int] = None
    currency: Optional[str] = None
    currency_raw: Optional[str] = Field(default=None, alias="currencyRaw")
    description: Optional[str] = None
    main_image: Optional[str] = Field(default=None, alias="mainImage")
    images: List[str] = Field(default_factory=list)


class SavedSearchRenderedPageResult(BaseModel):
    html: str
    provider: SavedSearchRenderProvider
    status_code: Optional[int] = Field(default=None, alias="statusCode")
    elapsed_ms: int = Field(default=0, alias="elapsedMs")
    action_statuses: List[str] = Field(default_factory=list, alias="actionStatuses")
    action_error: Optional[str] = Field(default=None, alias="actionError")
    network_capture_count: int = Field(default=0, alias="networkCaptureCount")
    network_captures: List[SavedSearchRenderedNetworkCapture] = Field(
        default_factory=list, alias="networkCaptures"
    )
    product_list_items: List[SavedSearchRenderedProductListItem] = Field(
        default_factory=list, alias="productListItems"
    )


class SavedSearchUrlDiagnostic(BaseModel):
    url: str
    resolved_url: Optional[str] = Field(default=None, alias="resolvedUrl")
    extraction_strategy: Optional[str] = Field(default=None, alias="extractionStrategy")
    candidates_before: Optional[int] = Field(default=None, alias="candidatesBefore")
    candidates_after: Optional[int] = Field(default=None, alias="candidatesAfter")
    candidate_url_pattern_include_count: int = Field(
        default=0, alias="candidateUrlPatternIncludeCount"
    )
    candidate_url_pattern_exclude_count: int = Field(
        default=0, alias="candidateUrlPatternExcludeCount"
    )
    dropped_by_missing_include_all_match: int = Field(
        default=0, alias="droppedByMissingIncludeAllMatch"
    )
    dropped_by_exclude_substring: int = Field(
        default=0, alias="droppedByExcludeSubstring"
    )
    first_dropped_by_missing_include_substring: Optional[str] = Field(
        default=None, alias="firstDroppedByMissingIncludeSubstring"
    )
    first_dropped_by_exclude_substring: Optional[str] = Field(
        default=None, alias="firstDroppedByExcludeSubstring"
    )
    render_fallback_attempted: bool = Field(
        default=False, alias="renderFallbackAttempted"
    )
    render_fallback_used: bool = Field(default=False, alias="renderFallbackUsed")
    render_fallback_provider: Optional[SavedSearchRenderProvider] = Field(
        default=None, alias="renderFallbackProvider"
    )
    render_fallback_trigger_policy: Optional[SavedSearchRenderTriggerPolicy] = Field(
        default=None, alias="renderFallbackTriggerPolicy"
    )
    render_fallback_http_status: Optional[int] = Field(
        default=None, alias="renderFallbackHttpStatus"
    )
    render_fallback_error: Optional[str] = Field(
        default=None, alias="renderFallbackError"
    )
    product_list_count: int = Field(default=0, alias="productListCount")
    product_list_mapped_count: int = Field(default=0, alias="productListMappedCount")
    merged_candidate_count: int = Field(default=0, alias="mergedCandidateCount")
    fetched: int = 0
    parsed: int = 0
    deduped: int = 0
    error: Optional[str] = None


class SavedSearchIngestResult(BaseModel):
    source_name: str = Field(alias="sourceName")
    source_urls: List[str] = Field(default_factory=list, alias="sourceUrls")
    candidates: List[SavedSearchCandidate] = Field(default_factory=list)
    samples: List[SavedSearchCandidate] = Field(default_factory=list)
    fetched: int = 0
    parsed: int = 0
    deduped: int = 0
    url_diagnostics: List[SavedSearchUrlDiagnostic] = Field(
        default_factory=list, alias="urlDiagnostics"
    )
