import asyncio
import csv
from datetime import datetime
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List

import pandas as pd

from fraudcrawler.settings import ROOT_DIR
from fraudcrawler.base.base import (
    Language,
    Location,
    Deepness,
    Host,
    ProductItem,
)
from fraudcrawler.base.orchestrator import Orchestrator
from fraudcrawler.scraping.search import Searcher, SearchEngineName
from fraudcrawler.scraping.enrich import Enricher
from fraudcrawler.scraping.url import URLCollector
from fraudcrawler.scraping.zyte import ZyteAPI
from fraudcrawler.processing.processor import Processor


logger = logging.getLogger(__name__)

_RESULTS_DIR = ROOT_DIR / "data" / "results"


class Results(BaseModel):
    """The results of the product search."""

    search_term: str
    filename: Path | None = None


class FraudCrawlerClient(Orchestrator):
    """The main client for FraudCrawler product search and analysis.

    This client orchestrates the complete pipeline: search, deduplication, context extraction,
    processing (classification), and result collection. It inherits from Orchestrator and adds
    result management and persistence functionality.
    """

    _FILENAME_TEMPLATE = "{search_term}_{language}_{location}_{timestamp}.csv"

    def __init__(
        self,
        searcher: Searcher,
        enricher: Enricher,
        url_collector: URLCollector,
        zyteapi: ZyteAPI,
        processor: Processor,
    ):
        """Initializes FraudCrawlerClient.

        Args:
            searcher: Client for searching step.
            enricher: Client for enrichment step.
            url_collector: Client for deduplication.
            zyteapi: Client for metadata extraction.
            processor: Client for product classification.
        """
        super().__init__(
            searcher=searcher,
            enricher=enricher,
            url_collector=url_collector,
            zyteapi=zyteapi,
            processor=processor,
        )

        self._results_dir = _RESULTS_DIR
        if not self._results_dir.exists():
            self._results_dir.mkdir(parents=True)
        self._results: List[Results] = []

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

        # Save the DataFrame to a CSV file
        filename = self._results[-1].filename
        df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Results saved to {filename}")

    async def run(
        self,
        search_term: str,
        search_engines: List[SearchEngineName],
        language: Language,
        location: Location,
        deepness: Deepness,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
        previously_collected_urls: List[str] | None = None,
    ) -> None:
        """Runs the pipeline steps: srch, deduplication, context extraction, processing, and collect the results.

        Args:
            search_term: The search term for the query.
            search_engines: The list of search engines to use for the search query.
            language: The language to use for the query.
            location: The location to use for the query.
            deepness: The search depth and enrichment details.
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
            previously_collected_urls: The urls that have been collected previously and are ignored.
        """
        # Handle results files
        timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
        filename = self._results_dir / self._FILENAME_TEMPLATE.format(
            search_term=search_term,
            language=language.code,
            location=location.code,
            timestamp=timestamp,
        )
        self._results.append(Results(search_term=search_term, filename=filename))

        # Run the pipeline by calling the orchestrator's run method
        await super().run(
            search_term=search_term,
            search_engines=search_engines,
            language=language,
            location=location,
            deepness=deepness,
            marketplaces=marketplaces,
            excluded_urls=excluded_urls,
            previously_collected_urls=previously_collected_urls,
        )

    def load_results(self, index: int = -1) -> pd.DataFrame:
        """Loads the results from the saved .csv files.

        Args:
            index: The index of the results to load (`incex=-1` are the results for the most recent run).
        """

        results = self._results[index]
        if (filename := results.filename) is None:
            raise ValueError("filename not found (is None)")

        return pd.read_csv(filename)

    def print_available_results(self) -> None:
        """Prints the available results."""
        n_res = len(self._results)
        for i, res in enumerate(self._results):
            print(f"index={-n_res + i}: {res.search_term} - {res.filename}")
