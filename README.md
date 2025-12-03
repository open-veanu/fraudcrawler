# fraudcrawler

![CI Status](https://github.com/open-veanu/fraudcrawler/workflows/CI/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PyPI](https://img.shields.io/pypi/v/fraudcrawler.svg)

Fraudcrawler is an intelligent **market monitoring** tool that searches the web for products, extracts product details, and classifies them using LLMs. It combines search APIs, web scraping, and AI to automate product discovery and relevance assessment.

## Features

- **Asynchronous pipeline** - Products move through search, extraction, and classification stages independently
- **Multiple search engines** - Google Search, Google Shopping, and more...
- **Search term enrichment** - Automatically find related terms and expand your search
- **Product extraction** - Get structured product data via Zyte API
- **LLM classification** - Assess product relevance using OpenAI API with custom prompts
- **Marketplace filtering** - Focus searches on specific domains
- **Deduplication** - Avoid reprocessing previously collected URLs
- **CSV export** - Results saved with timestamps for easy tracking

## Prerequisites

- Python 3.11 or higher
- API keys for:
  - **SerpAPI** - Google search results
  - **Zyte API** - Product data extraction
  - **OpenAI API** - Product classification
  - **DataForSEO** (optional) - Search term enrichment

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install fraudcrawler
```

**Using Poetry:**
```bash
poetry install
```

## Configuration

Create a `.env` file with your API credentials (see `.env.example` for template):

```bash
SERPAPI_KEY=your_serpapi_key
ZYTEAPI_KEY=your_zyte_key
OPENAIAPI_KEY=your_openai_key
DATAFORSEO_USER=your_user  # optional
DATAFORSEO_PWD=your_pwd    # optional
```

## Usage

### Basic Configuration
For a complete working example, see `fraudcrawler/launch_demo_pipeline.py`. After setting up the necessary parameters you can launch and analyse the results with:
```python
# Run pipeline
await client.run(
    search_term=search_term,
    search_engines=search_engines,
    language=language,
    location=location,
    deepness=deepness,
    excluded_urls=excluded_urls,
)

# Load results
df = client.load_results()
print(df.head())
```

### Advanced Configuration

**Search term enrichment** - Find and search related terms:
```python
from fraudcrawler import Enrichment

deepness.enrichment = Enrichment(
    additional_terms=5,
    additional_urls_per_term=10
)
```

**Marketplace filtering** - Focus on specific domains:
```python
from fraudcrawler import Host

marketplaces = [
    Host(name="International", domains="zavamed.com,apomeds.com"),
    Host(name="National", domains="netdoktor.ch,nobelpharma.ch"),
]

await client.run(..., marketplaces=marketplaces)
```

**Exclude domains** - Exclude specific domains from your results:
```python
excluded_urls = [
    Host(name="Compendium", domains="compendium.ch"),
]

await client.run(..., excluded_urls=excluded_urls)
```

**Skip previously collected URLs**:
```python
previously_collected_urls = [
    "https://example.com/product1",
    "https://example.com/product2",
]

await client.run(..., previously_collected_urls=previously_collected_urls)
```

**View all results** from a client instance:
```python
client.print_available_results()
```

## Output

Results are saved as CSV files in `data/results/` with the naming pattern:

```
<search_term>_<language_code>_<location_code>_<timestamp>.csv
```

Example: `sildenafil_de_ch_20250115143022.csv`

The CSV includes product details, URLs, and classification scores from your workflows.

## Development

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Architecture

Fraudcrawler uses an asynchronous pipeline where products can be at different processing stages simultaneously. Product A might be in classification while Product B is still being scraped. This is enabled by async workers for each stage (Search, Context Extraction, Processing) using `httpx.AsyncClient`.

![Async Setup](https://github.com/open-veanu/fraudcrawler/raw/master/docs/assets/images/Fraudcrawler_Async_Setup.svg)

For more details on the async design, see the [httpx documentation](https://www.python-httpx.org/api/#asyncclient).
