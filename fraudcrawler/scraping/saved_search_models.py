from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SavedSearchRenderRequestHeaders(BaseModel):
    referer: Optional[str] = None


class SavedSearchRenderOptionsConfig(BaseModel):
    javascript: Optional[bool] = None
    include_iframes: Optional[bool] = Field(default=None, alias="includeIframes")
    request_headers: Optional[SavedSearchRenderRequestHeaders] = Field(
        default=None, alias="requestHeaders"
    )
    actions: Optional[List[dict[str, Any]]] = None
    network_capture: Optional[List[dict[str, Any]]] = Field(
        default=None, alias="networkCapture"
    )


class WebsiteSourceSearchableUrl(BaseModel):
    model_config = ConfigDict(extra="allow")

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
            raise ValueError("Website-source filterUrl must not be empty.")
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


class WebsiteSourceUrlTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(alias="baseUrl")
    searchable_urls: List[WebsiteSourceSearchableUrl] = Field(alias="searchableUrls")

    @field_validator("base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, value: Any) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("Website-source baseUrl must not be empty.")
        return normalized


class WebsiteSourceFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    mode: str = "per_source_all_urls"
    render_options: SavedSearchRenderOptionsConfig = Field(alias="renderOptions")


class WebsiteSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    urls: List[WebsiteSourceUrlTemplate]
    search_filter_config: Optional[WebsiteSourceFilterConfig] = Field(
        default=None, alias="searchFilterConfig"
    )
