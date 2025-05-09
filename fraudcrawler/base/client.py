import asyncio
import csv
from datetime import datetime
import hashlib
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List

import pandas as pd

from fraudcrawler.settings import ROOT_DIR
from fraudcrawler.base.base import Setup, Language, DSsettings, Location, Deepness, Host, Prompt
from fraudcrawler.base.orchestrator import Orchestrator, ProductItem

logger = logging.getLogger(__name__)

_RESULTS_DIR = ROOT_DIR / "data" / "results"


class Results(BaseModel):
    """The results of the product search."""

    search_term: str
    filename: Path | None = None


class FraudCrawlerClient(Orchestrator):
    """The main client for FraudCrawler."""

    _filename_template = "{filename_hash}_{search_term}_{language}_{location}_{timestamp}.csv"

    def __init__(self):
        setup = Setup()
        super().__init__(
            serpapi_key=setup.serpapi_key,
            dataforseo_user=setup.dataforseo_user,
            dataforseo_pwd=setup.dataforseo_pwd,
            zyteapi_key=setup.zyteapi_key,
            openaiapi_key=setup.openaiapi_key,
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
        search_term: str,
        language: Language,
        location: Location,
        deepness: Deepness,
        prompts: List[Prompt],
        ds_settings: DSsettings,
        marketplaces: List[Host] | None = None,
        excluded_urls: List[Host] | None = None,
    ) -> None:
        """Runs the pipeline steps: serp, enrich, zyte, process, and collect the results.

        Args:
            search_term: The search term for the query.
            language: The language to use for the query.
            location: The location to use for the query.
            deepness: The search depth and enrichment details.
            prompts: The list of prompts to use for classification.
            ds_settings: Data Science setting for creating and testing of benchmarks.
            marketplaces: The marketplaces to include in the search.
            excluded_urls: The URLs to exclude from the search.
        """
        timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
        run_hash = hashlib.sha256("{search_term}_{language}_{location}_{timestamp}.csv".encode()).hexdigest()[:8]
        # Filename to be saved for results depends on if data science validations 
        if not ds_settings.use_cached_ds_data: 
            filename = self._results_dir / self._filename_template.format(
                filename_hash = run_hash,
                search_term=search_term,
                language=language.code,
                location=location.code,
                timestamp=timestamp,
            )
        else:
            filename = ds_settings.cached_filename.removesuffix(".csv") + "_" + datetime.today().strftime("%Y%m%d%H%M%S") + '.csv'
            filename = self._results_dir / filename

        self._results.append(Results(search_term=search_term, filename=filename))

        asyncio.run(
            super().run(
                search_term=search_term,
                language=language,
                location=location,
                deepness=deepness,
                ds_settings=ds_settings,
                prompts=prompts,
                timestamp=timestamp,
                marketplaces=marketplaces,
                excluded_urls=excluded_urls,
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
