# CLAUDE.md

FraudCrawler open-source package. FraudCrawler is a **market monitoring** tool that searches the web for products, and classifies them. It combines search APIs, web scraping, and AI to automate product discovery and relevance assessment.

## Architecture

### Orchestrator
The core object (@fraucawler/base/orchestrator.py) that orchestrates the pipeline with 5 main steps:
1. Search urls
2. Deduplicate urls (each url is a product)
3. Extract product details for each url
4. Compute relevance for each product based on details
5. Collect and aggregate the products

All of the steps are handled async. Step 5 is abstract and can be changed according to the usage of the package.

### Local execution
The FraudCrawlerClient (@fraudcrawler/base/client.py) implements step 5 by aggregating the products into a pandas.Dataframe. The file @launch_demo_pipeline.py gives gives an example that can be run locally.

### FraudCrawler Backend
When wrapping fraudcrawler inside a backend, it might implement step 5 by storing products into a database.

### Search
Searching products and extracting context is handled by Searcher (@fraudcrawler/scraping/search.py).

### Processing
Processing the products and assessing their relevance is handled by Processor (@fraudcrawler/processing/base.py).

### Caching
Caching of http requests is handled by RedisCacher (@fraudcrawler/cache/cacher.py)

### Parametrization
The package is parametrized inside a settings module (@fraudcrawler/settings.py).

## CI/CD
Type checks, vulnerability checks, linting and unittests are set up inside a github workflow (@.github/workflows/ci.yml)
