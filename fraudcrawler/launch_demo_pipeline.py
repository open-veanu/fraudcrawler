import logging

from fraudcrawler import FraudCrawlerClient, Language, DSsettings, Location, Deepness, Prompt

LOG_FMT = "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s"
LOG_LVL = "INFO"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=LOG_FMT, level=LOG_LVL, datefmt=DATE_FMT)


def main():
    # Setup the client
    client = FraudCrawlerClient()

    # Setup the search
    search_term = "Nitrobenzin"
    language = Language(name="German")
    location = Location(name="Switzerland")
    deepness = Deepness(num_results=100)
    prompts = [
        Prompt(
            name="relevance",
            context="This organization is interested products sold in an online shop regarding health. The organization monitors "
                    "health, radiation, environmental health risks and ensures food and chemical safety, among other tasks. "
                    "They only want to know about relevant sold products for their health context, such as but not limited to:"
                    "- medicines, supplements, cosmetics"
                    "- medical devices"
                    "- chemicals, fertilizers"
                    "- food and beverages"
                    "- medical and radiation protection"
                    "they are only interested in products sold in a shop, not in pure information pages",
            system_prompt=(
                "You are a helpful and intelligent assistant. Your task is to classify any given product "
                "as either relevant (1) or not relevant (0), strictly based on the context and product details provided by the user. "
                "Products should be classified as relevant (1) if they are explicitly sold. A description of a product is "
                "not enough to classify it as relevant. A good indicator is if the text mentions prices, shipping costs, a quantity" 
                " in gram, kg or another measurement unit, or descriptions of pills, liquid doses or tablets."
                "You must consider all aspects of the given context and make a binary decision accordingly. "
                "If the product aligns with the user's needs, classify it as 1 (relevant); otherwise, classify it as 0 (not relevant). "
                "Respond only with the number 1 or 0."
            ),
            allowed_classes=[0, 1],
        ),
        Prompt(
            name="seriousness",
            context="This organization is interested in checking for products relevant to health. Any product such as "
                    "a medicine, a supplement, a chemical, a fertilizer, a cosmetic, etc are relevant to them",
            system_prompt=(
                "You are an intelligent and discerning assistant. Your task is to classify each item as either "
                "relevant content (1) or not relevant content (0). To make this distinction, consider the following criteria: \n"
                "    1. Relevant Content (1): Classify as 1 if the result clearly indicates content relevant to health, such as but not limited to:\n"
                "        - Medicines, supplements, cosmetics"
                "        - Chemicals, fertilizers"
                "        - Medical devices"
                "        - Food and beverages"
                "        - Medical and radiation protection"
                "    2. Not Relevant Content (0): Classify as 0 if the result is not relevant to the organization. For example: \n"
                "        - Books: They are never relevant, classify as 0.\n"
                "        - An article that seems describe a book is also never relevant.\n"
                "        - Videos: Not relevant, classify as 0.\n"
                "        - Advertisements: Promotional content that doesn't directly sell a product.\n"
                "        - Information pages: If a page offers information about a substance or product, but doesn't sell it, classify as 0.\n"
                "        - Companies and Services: Names and descriptions of companies or services related to the product but not the product itself.\n"
                "        - Related Topics/Content: Any text or media that discusses or elaborates on the topic without offering a tangible product for sale.\n"
                "Make your decision based solely on the context and details provided in the search result. Respond only with the number 1 or 0."
            ),
            allowed_classes=[0, 1],
        ),
    ]
    # # Optional: Add tern ENRICHMENT
    from fraudcrawler import Enrichment

    deepness.enrichment = Enrichment(additional_terms=10, additional_urls_per_term=20)

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

    # Data Science Settings
    ds_settings = DSsettings(
                            dataset_creation=True,
                            use_cached_ds_data =False,
                            cached_filename=None #59ab8652_Borax_de_ch_20250502140755_labeled.xlsx or None
                            )

    # Execute the pipeline
    client.execute(
        search_term=search_term,
        language=language,
        location=location,
        deepness=deepness,
        prompts=prompts,
        ds_settings=ds_settings,
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
