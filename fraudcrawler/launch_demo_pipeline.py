import logging

from fraudcrawler import FraudCrawlerClient, Language, Location, Deepness, Prompt

LOG_FMT = "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s"
LOG_LVL = "INFO"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=LOG_FMT, level=LOG_LVL, datefmt=DATE_FMT)


def main():
    # Setup the client
    client = FraudCrawlerClient()

    # Setup the search
    search_term = "Kühlschrank"
    language = Language(name="German")
    location = Location(name="Switzerland")
    deepness = Deepness(num_results=10)
    prompts = [
        Prompt(
            name="availability",
            system_prompt=(
                "You are a helpful and intelligent assistant helping an organization that is interested in checking the availability of certain products."
                "Your task is to classify any given product as either available (1) or not available (0), strictly based on the context and product details provided by the user. "
                "You must consider all aspects of the given context and make a binary decision accordingly. "
                "If the product can be purchased, added to a shopping basket, delivered, or is listed as available in any form, classify it as 1 (available); "
                "if there is any mention of out of stock, not available, no longer shippable, or similar, classify it as 0 (not available). "
                "Respond only with the number 1 or 0."
            ),
            product_item_fields=["product_name", "html_clean"],
            allowed_classes=[0, 1],
        ),
        # Prompt(
        #     name="seriousness",
        #     system_prompt=(
        #         "You are a helpful and intelligent assistant helping an organization that is interested in checking the energy efficiency of certain devices. "
        #         "Your task is to classify each item as either a product for sale (1) or not a product for sale (0). To make this distinction, consider the following criteria: \n"
        #         "    1 Product for Sale (1): Classify as 1 if the result clearly indicates an item available for purchase, typically found  "
        #         "within an online shop or marketplace.\n"
        #         "    2 Not a Product for Sale (0): Classify as 0 if the result is unrelated to a direct purchase of a product. This includes items such as: \n"
        #         "        - Books and Videos: These may be available for sale, but if they are about or related to the searched product rather than being the "
        #         "exact product itself, classify as 0.\n"
        #         "        - Advertisements: Promotional content that doesn't directly sell a product.\n"
        #         "        - Companies and Services: Names and descriptions of companies or services related to the product but not the product itself.\n"
        #         "        - Related Topics/Content: Any text or media that discusses or elaborates on the topic without offering a tangible product for sale.\n"
        #         "Make your decision based solely on the context and details provided in the search result. Respond only with the number 1 or 0."
        #     ),
        #     product_item_fields=["product_name", "product_description"],
        #     allowed_classes=[0, 1],
        # ),
    ]
    # # Optional: Add tern ENRICHEMENT
    # from fraudcrawler import Enrichment

    # deepness.enrichment = Enrichment(additional_terms=10, additional_urls_per_term=20)

    # # Optional: Add MARKETPLACES and EXCLUDED_URLS
    # from fraudcrawler import Host

    # marketplaces = [
    #     Host(name="International", domains="zavamed.com,apomeds.com"),
    #     Host(name="National", domains="netdoktor.ch, nobelpharma.ch")
    # ]
    # excluded_urls = [
    #     Host(name="Digitec", domains="digitec.ch"),
    #     Host(name="Brack", domains="brack.ch"),
    # ]

    # Execute the pipeline
    client.execute(
        search_term=search_term,
        language=language,
        location=location,
        deepness=deepness,
        prompts=prompts,
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


if __name__ == "__main__":
    main()
