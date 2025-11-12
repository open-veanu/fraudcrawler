import asyncio
import csv
from datetime import datetime
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Self

import httpx
import pandas as pd

from fraudcrawler.settings import ROOT_DIR
from fraudcrawler.base.base import (
    Setup,
    ProductItem,
)
from fraudcrawler.base.orchestrator import Orchestrator
from fraudcrawler.scraping.arguments import ScrapingArgs
from fraudcrawler.processing.arguments import ProcessingArgs
from fraudcrawler.processing.processor import OpenAIChat, Processor

logger = logging.getLogger(__name__)

_RESULTS_DIR = ROOT_DIR / "data" / "results"


class Results(BaseModel):
    """The results of the product search."""

    search_term: str
    filename: Path | None = None


class OpenAIChatSetup(BaseModel):
    name: str
    model: str
    product_item_fields: List[str]
    system_prompt: str
    allowed_classes: List[int]


class FraudCrawlerProcessor(Processor):
    """Sets up OpenAIChat workflows for a given set of system prompts."""

    def __init__(
            self,
            http_client: httpx.AsyncClient,
            openaiapi_key: str
        ):
        self._openaiapi_key = openaiapi_key
        super().__init__(http_client=http_client)
    
    def _setup_workflows(self, http_client: httpx.AsyncClient):
        openai_chat_setups = [
            {
                "name": "availability",
                "model": "gpt-4o",
                "product_item_fields": ["product_name", "html_clean"],
                "system_prompt": (
                    "You are a helpful and intelligent assistant helping an organization that is interested in checking the availability of certain products."
                    "Your task is to classify any given product as either available (1) or not available (0), strictly based on the context and product details provided by the user. "
                    "You must consider all aspects of the given context and make a binary decision accordingly. "
                    "If the product can be purchased, added to a shopping basket, delivered, or is listed as available in any form, classify it as 1 (available); "
                    "if there is any mention of out of stock, not available, no longer shippable, or similar, classify it as 0 (not available). "
                    "Respond only with the number 1 or 0."
                ),
                "allowed_classes": [0, 1],
            },
            {
                "name": "seriousness",
                "model": "gpt-4o",
                "product_item_fields": ["product_name", "product_description"],
                "system_prompt": (
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
                ),
                "allowed_classes": [0, 1],
            },
        ]
        return [OpenAIChat(http_client=http_client, api_key=self._openaiapi_key, **stps) for stps in openai_chat_setups]


class FraudCrawlerClient(Orchestrator):
    """The main client for FraudCrawler."""

    _filename_template = "{search_term}_{language}_{location}_{timestamp}.csv"

    def __init__(self):
        setup = Setup()
        super().__init__(
            serpapi_key=setup.serpapi_key,
            dataforseo_user=setup.dataforseo_user,
            dataforseo_pwd=setup.dataforseo_pwd,
            zyteapi_key=setup.zyteapi_key,
        )
        self._openaiapi_key = setup.openaiapi_key

        self._results_dir = _RESULTS_DIR
        if not self._results_dir.exists():
            self._results_dir.mkdir(parents=True)
        self._results: List[Results] = []


    def _setup_processor(self, http_client: httpx.AsyncClient) -> Processor:
        return FraudCrawlerProcessor(http_client=http_client, openaiapi_key=self._openaiapi_key)

    async def __aenter__(self) -> Self:
        await super().__aenter__()  # let base set itself up
        return self  # so `async with FraudCrawlerClient()` gives you this instance

    async def __aexit__(self, *args, **kwargs) -> None:
        await super().__aexit__(*args, **kwargs)

    async def _collect_results(
        self, queue_in: asyncio.Queue[ProductItem | None]
    ) -> None:
        """Collects the results from the given queue_in and saves it as csv.

        Args:
            queue_in: The input queue containing the results.
        """
        products = []
        while True:
            product = await queue_in.get()
            if product is None:
                queue_in.task_done()
                break

            products.append(product.model_dump())
            queue_in.task_done()

        # Convert the list of products to a DataFrame
        df = pd.json_normalize(products)
        cols = [c.split(".")[-1] for c in df.columns]
        if len(cols) != len(set(cols)):
            logger.error("Duplicate columns after json_normalize.")
        else:
            df.columns = cols

        # Save the DataFrame to a CSV file
        filename = self._results[-1].filename
        df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Results saved to {filename}")

    def execute(
        self,
        scrp_args: ScrapingArgs,
        proc_args: ProcessingArgs,
    ) -> None:
        """Runs the pipeline steps: srch, deduplication, context extraction, processing, and collect the results.

        Args:
            scrp_args: Sets up the scraping pipeline step.
            proc_args: Arguments for setting up the processing workflows.
        """

        # Handle results files
        timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
        filename = self._results_dir / self._filename_template.format(
            search_term=scrp_args.search_term,
            language=scrp_args.language.code,
            location=scrp_args.location.code,
            timestamp=timestamp,
        )
        self._results.append(Results(search_term=scrp_args.search_term, filename=filename))

        # Run the pipeline by calling the orchestrator's run method
        async def _run(*args, **kwargs):
            async with self:
                return await super(FraudCrawlerClient, self).run(*args, **kwargs)

        asyncio.run(
            _run(
                scrp_args=scrp_args,
                proc_args=proc_args,
            )
        )

    def load_results(self, index: int = -1) -> pd.DataFrame:
        """Loads the results from the saved .csv files.

        Args:
            index: The index of the results to load (`incex=-1` are the results for the most recent run).
        """

        results = self._results[index]
        return pd.read_csv(results.filename)

    def print_available_results(self) -> None:
        """Prints the available results."""
        n_res = len(self._results)
        for i, res in enumerate(self._results):
            print(f"index={-n_res + i}: {res.search_term} - {res.filename}")
