import asyncio
import logging
from typing import Any, Sequence

from fraudcrawler.base.base import Setup
from fraudcrawler import (
    FraudCrawlerClient,
    HttpxAsyncClient,
    Searcher,
    Enricher,
    URLCollector,
    ZyteAPI,
    SearchEngineName,
    Language,
    Location,
    Deepness,
    Processor,
    Workflow,
    OpenAIClassification,
    WebsiteSource,
)
from fraudcrawler.scraping.search import WebsiteSearch
from fraudcrawler.scraping.saved_search_models import (
    WebsiteSourceFilterConfig,
    WebsiteSourceUrlTemplate,
)

LOG_FMT = "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s"
LOG_LVL = "INFO"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
REDIS_USE_CACHE = True
SETUP = Setup()  # type: ignore[call-arg]
logging.basicConfig(format=LOG_FMT, level=LOG_LVL, datefmt=DATE_FMT)


def _build_website_source_profile(
    *,
    name: str,
    base_url: str,
    searchable_urls: list[dict[str, Any]],
    render_fallback: dict[str, Any] | None = None,
) -> WebsiteSource:
    filter_config: dict[str, Any] = {
        "version": 1,
        "mode": "per_source_all_urls",
    }
    if render_fallback is not None:
        filter_config["renderFallback"] = render_fallback

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


GALAXUS_WEBSITE_SOURCE_PROFILE = _build_website_source_profile(
    name="Boost Galaxus",
    base_url="https://www.galaxus.ch/",
    searchable_urls=[
        {
            "filterUrl": "de/search?q={search_term}&filter=21826=892",
            "includeSubstrings": ["/product/"],
            "excludeSubstrings": [],
        }
    ],
    render_fallback={
        "enabled": True,
        "provider": "zyte",
        "triggerPolicy": "on_zero_candidates",
        "javascript": True,
        "includeIframes": False,
        "requestHeaders": {"referer": "https://www.galaxus.ch/"},
        "actions": [],
        "networkCapture": [],
    },
)


FUST_WEBSITE_SOURCE_PROFILE = _build_website_source_profile(
    name="Fust",
    base_url="https://www.fust.ch/",
    searchable_urls=[
        {
            "filterUrl": "search?q={search_term}:relevance:Energieeffizienzklasse:A",
            "includeSubstrings": ["/p/"],
            "excludeSubstrings": [],
        }
    ],
    render_fallback={
        "enabled": True,
        "provider": "zyte",
        "triggerPolicy": "on_zero_candidates",
        "javascript": True,
        "includeIframes": False,
        "requestHeaders": {"referer": "https://www.fust.ch/"},
        "actions": [],
        "networkCapture": [],
    },
)

FRANKENSPALTER_WEBSITE_SOURCE_PROFILE = _build_website_source_profile(
    name="Frankenspalter",
    base_url="https://www.frankenspalter.ch/",
    searchable_urls=[
        {
            "filterUrl": "de/catalogsearch/result/index/?q={search_term}&itg_202303151629545861692577=1818",
            "includeSubstrings": [],
            "excludeSubstrings": [],
        }
    ],
    # important: no render fallback because frankenspalter allows direct crawling.
)

DEFAULT_WEBSITE_SOURCE_PROFILE = GALAXUS_WEBSITE_SOURCE_PROFILE


def _setup_workflows(
    http_client: HttpxAsyncClient, redis_use_cache: bool
) -> Sequence[Workflow]:
    """Sets up the set of workflows to be run iteratively."""
    _AVAILABILITY_SYSTEM_PROMPT = (
        "You are a helpful and intelligent assistant helping an organization that is interested in checking the availability of certain products."
        "Your task is to classify any given product as either available (1) or not available (0), strictly based on the context and product details provided by the user. "
        "You must consider all aspects of the given context and make a binary decision accordingly. "
        "If the product can be purchased, added to a shopping basket, delivered, or is listed as available in any form, classify it as 1 (available); "
        "if there is any mention of out of stock, not available, no longer shippable, or similar, classify it as 0 (not available). "
        "Respond only with the number 1 or 0."
    )
    _SERIOUSNESS_SYSTEM_PROMPT = (
        "You are a helpful and intelligent assistant helping an organization that is interested in checking the energy efficiency of certain devices. "
        "Your task is to classify each item as either a product for sale (1) or not a product for sale (0). To make this distinction, consider the following criteria: \n"
        "    1 Product for Sale (1): Classify as 1 if the result clearly indicates an item available for purchase, typically found  "
        "within an online shop or marketplace.\n"
        "    2 Not a Product for Sale (0): Classify as 0 if the result is unrelated to a direct purchase of a product. This includes items such as: \n"
        "        - Books and Videos: These may be available for sale, but if they are about or related to the searched product rather than being the "
        "exact product itself, classify as 0.\n"
        "        - Advertisements: Promotional content that doesn't directly sell a product.\n"
        "        - Companies and Services: Names and descriptions of companies or services related to the product but not the product itself.\n"
        "        - Related Topics/Content: Any text or media that discusses or elaborates on the topic without offering a tangible product for sale.\n"
        "Make your decision based solely on the context and details provided in the search result. Respond only with the number 1 or 0."
    )
    return [
        OpenAIClassification(
            http_client=http_client,
            name="availability",
            api_key=SETUP.openaiapi_key,
            model="gpt-4o",
            product_item_fields=["product_name", "html_clean"],
            system_prompt=_AVAILABILITY_SYSTEM_PROMPT,
            allowed_classes=[0, 1],
            redis_use_cache=redis_use_cache,
        ),
        OpenAIClassification(
            http_client=http_client,
            name="seriousness",
            api_key=SETUP.openaiapi_key,
            model="gpt-4o",
            product_item_fields=["product_name", "product_description"],
            system_prompt=_SERIOUSNESS_SYSTEM_PROMPT,
            allowed_classes=[0, 1],
            redis_use_cache=redis_use_cache,
        ),
    ]


async def run(http_client: HttpxAsyncClient, search_term: str):
    # Setup the search
    search_engines = list(SearchEngineName)
    language = Language(name="German")
    location = Location(name="Switzerland")
    deepness = Deepness(num_results=10)

    # # Optional: Add term enrichment
    # from fraudcrawler import Enrichment

    # deepness.enrichment = Enrichment(additional_terms=10, additional_urls_per_term=20)

    # Optional: Add MARKETPLACES and EXCLUDED_URLS
    # from fraudcrawler import Host

    # marketplaces = [
    #     Host(name="International", domains="zavamed.com,apomeds.com"),
    #     # Host(name="National", domains="netdoktor.ch, nobelpharma.ch")
    # ]
    # excluded_urls = [
    #     Host(name="Digitec", domains="digitec.ch"),
    #     Host(name="Brack", domains="brack.ch"),
    # ]

    # Setup clients
    searcher = Searcher(
        http_client=http_client,
        serpapi_key=SETUP.serpapi_key,
        zyteapi_key=SETUP.zyteapi_key,
        redis_use_cache=REDIS_USE_CACHE,
    )
    enricher = Enricher(
        http_client=http_client,
        user=SETUP.dataforseo_user,
        pwd=SETUP.dataforseo_pwd,
        redis_use_cache=REDIS_USE_CACHE,
    )
    url_collector = URLCollector()
    zyteapi = ZyteAPI(
        http_client=http_client,
        api_key=SETUP.zyteapi_key,
        redis_use_cache=REDIS_USE_CACHE,
    )
    workflows = _setup_workflows(
        http_client=http_client, redis_use_cache=REDIS_USE_CACHE
    )
    processor = Processor(workflows=workflows)

    # Setup the client
    client = FraudCrawlerClient(
        searcher=searcher,
        enricher=enricher,
        url_collector=url_collector,
        zyteapi=zyteapi,
        processor=processor,
    )

    # Execute the pipeline
    await client.run(
        search_term=search_term,
        search_engines=search_engines,
        language=language,
        location=location,
        deepness=deepness,
        website_source_sources=[DEFAULT_WEBSITE_SOURCE_PROFILE],
        # marketplaces=marketplaces,
        # excluded_urls=excluded_urls,
    )

    # Show results
    print()
    title = "Available results"
    print(title)
    print("=" * len(title))
    client.print_available_results()
    print()
    title = f'Results for "{search_term.upper()}"'
    print(title)
    print("=" * len(title))
    df = client.load_results()
    print(f"Number of products found: {len(df)}")
    print()
    n_head = 10
    print(f"First {n_head} products are:")
    print(df.head(n=n_head))
    print()


async def run_website_source_demo(http_client: HttpxAsyncClient, search_term: str):
    website_search = WebsiteSearch(
        http_client=http_client,
        zyteapi_key=SETUP.zyteapi_key,
        redis_use_cache=REDIS_USE_CACHE,
    )
    result = await website_search.ingest_source(
        source=DEFAULT_WEBSITE_SOURCE_PROFILE,
        search_term=search_term,
        max_items=50,
    )
    print()
    title = f'Website-source results for "{search_term.upper()}"'
    print(title)
    print("=" * len(title))
    print(f"Source: {result.source_name}")
    print(f"Resolved URLs: {len(result.source_urls)}")
    print(f"Candidates: {len(result.candidates)}")
    print("Sample URLs:")
    for sample in result.samples[:10]:
        print(f"- {sample.url}")
    print()
    print("URL diagnostics:")
    for diag in result.url_diagnostics:
        print(
            (
                f"- resolved={diag.resolved_url} | "
                f"strategy={diag.extraction_strategy} | "
                f"fallback_attempted={diag.render_fallback_attempted} | "
                f"fallback_used={diag.render_fallback_used} | "
                f"fallback_provider={diag.render_fallback_provider} | "
                f"fallback_http={diag.render_fallback_http_status} | "
                f"product_list_count={diag.product_list_count} | "
                f"product_list_mapped={diag.product_list_mapped_count} | "
                f"merged_candidates={diag.merged_candidate_count} | "
                f"fallback_error={diag.render_fallback_error}"
            )
        )


async def main(search_term: str):
    async with HttpxAsyncClient() as http_client:
        await run(http_client=http_client, search_term=search_term)


async def main_website_source_demo(search_term: str):
    async with HttpxAsyncClient() as http_client:
        await run_website_source_demo(http_client=http_client, search_term=search_term)


if __name__ == "__main__":
    # asyncio.run(main(search_term="Kühlschrank"))
    asyncio.run(main_website_source_demo(search_term="Kühlschrank"))
