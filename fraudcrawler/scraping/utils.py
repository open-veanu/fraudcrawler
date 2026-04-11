from typing import Any

from fraudcrawler.scraping.saved_search_models import (
    WebsiteSource,
    WebsiteSourceFilterConfig,
    WebsiteSourceUrlTemplate,
)


def build_website_source_profile(
    *,
    name: str,
    base_url: str,
    searchable_urls: list[dict[str, Any]],
    render_options: dict[str, Any],
) -> WebsiteSource:
    filter_config: dict[str, Any] = {
        "version": 1,
        "mode": "per_source_all_urls",
        "renderOptions": render_options,
    }

    return WebsiteSource(
        name=name,
        urls=[
            WebsiteSourceUrlTemplate.model_validate(
                {
                    "baseUrl": base_url,
                    "searchableUrls": searchable_urls,
                }
            )
        ],
        searchFilterConfig=WebsiteSourceFilterConfig.model_validate(filter_config),
    )
