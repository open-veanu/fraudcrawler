import logging

from fraudcrawler import (
    FraudCrawlerClient,
    SearchEngineName,
    Language,
    Location,
    Deepness,
    ScrapingArgs,
    ProcessingArgs,
)

LOG_FMT = "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s"
LOG_LVL = "INFO"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=LOG_FMT, level=LOG_LVL, datefmt=DATE_FMT)


def _get_scraping_args(search_term: str) -> ScrapingArgs:
    # Setup the search
    language = Language(name="German")
    location = Location(name="Switzerland")
    deepness = Deepness(num_results=10)

    # # Optional: Add tern ENRICHEMENT
    # from fraudcrawler import Enrichment

    # deepness.enrichment = Enrichment(additional_terms=10, additional_urls_per_term=20)

    # Optional: Add MARKETPLACES and EXCLUDED_URLS
    from fraudcrawler import Host

    # marketplaces = [
    #     Host(name="International", domains="zavamed.com,apomeds.com"),
    #     # Host(name="National", domains="netdoktor.ch, nobelpharma.ch")
    # ]
    excluded_urls = [
        Host(name="Digitec", domains="digitec.ch"),
        Host(name="Brack", domains="brack.ch"),
    ]
    return ScrapingArgs(
        search_term=search_term,
        search_engines=list(SearchEngineName),
        language=language,
        location=location,
        deepness=deepness,
        # marketplaces=marketplaces,
        excluded_urls=excluded_urls,
    )


def _get_processing_args() -> ProcessingArgs:
    return {}


def main(search_term: str):
    # Get configs
    scrp_args = _get_scraping_args(search_term=search_term)
    proc_args = _get_processing_args()

    # Setup the client
    client = FraudCrawlerClient(scrp_args=scrp_args, proc_args=proc_args)

    # Execute the pipeline
    client.execute()

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


if __name__ == "__main__":
    main(search_term="Kaffeebohnen")
