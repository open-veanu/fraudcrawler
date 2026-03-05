import asyncio

import pandas as pd
import pytest

from fraudcrawler.base.base import ProductItem, WebsiteSourceMetadata
from fraudcrawler.base.client import FraudCrawlerClient, Results


@pytest.mark.asyncio
async def test_collect_results_excludes_raw_html_field(tmp_path):
    client = FraudCrawlerClient(
        searcher=object(),  # type: ignore[arg-type]
        enricher=object(),  # type: ignore[arg-type]
        url_collector=object(),  # type: ignore[arg-type]
        zyteapi=object(),  # type: ignore[arg-type]
        processor=object(),  # type: ignore[arg-type]
    )
    output_file = tmp_path / "results.csv"
    client._results = [Results(search_term="test", filename=output_file)]

    queue: asyncio.Queue[ProductItem | None] = asyncio.Queue()
    await queue.put(
        ProductItem(
            search_term="mini fridge",
            search_term_type="initial",
            url="https://shop.test/p/123",
            url_resolved="https://shop.test/p/123",
            search_engine_name="website_source",
            domain="shop.test",
            product_name="Mini Fridge 120L",
            html="<html><body>secret payload</body></html>",
            html_clean="secret payload",
            website_source=WebsiteSourceMetadata(
                source_name="Boost Shop",
                resolved_url="https://shop.test/search?q=mini+fridge",
                render_http_status=200,
            ),
        )
    )
    await queue.put(None)

    await client._collect_results(queue_in=queue)

    df = pd.read_csv(output_file)
    assert "html" not in df.columns
    assert "html_clean" in df.columns
    assert "website_source.source_name" in df.columns
    assert "website_source.render_http_status" in df.columns
