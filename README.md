# open-veanu/fraudcrawler
Intelligent Market Monitoring

The pipeline for monitoring the market has the folling main steps:
1. search for a given term using SerpAPI
2. get product information using ZyteAPI
3. assess relevance of the found products using an OpenAI API

## Installation
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install fraudcrawler
```

## Usage
### `.env` file
Make sure to create an `.env` file with the necessary API keys and credentials (c.f. `.env.example` file).

### Run demo pipeline
```bash
python -m fraudcrawler.launch_demo_pipeline
```

### Customize the pipeline
Start by initializing the client
```python
from fraudcrawler import FraudCrawlerClient

# Initialize the client
client = FraudCrawlerClient()
```

For setting up the search we need 5 main objects.

#### `search_term: str`
The search term for the query (similar to search terms used within major search providers).

#### `language: Language`
The language used in SerpAPI ('hl' parameter), as well as for the optional search term enrichement (e.g. finding similar and related search terms). `language=Language('German')` creates an object having a language name and a language code as: `Language(name='German', code='de')`.

#### `location: Location`
The location used in SerpAPI ('gl' parameter). `location=Location('Switzerland')` creates an object having a location name and a location code as `Location(name='Switzerland', code='ch')`.

#### `deepness: Deepness`
Defines the search depth with the number of results to retrieve and optional enrichment parameters.

#### `prompts: List[Prompt]`
The list of prompts to classify a given product with (multiple) LLM calls. Each prompt object has a `name`, a `context` (used for defining the user prompt), a `system_prompt` (for defining the classification task), `allowed_classes` (a list of possible classes) and optionally `default_if_missing` (a default class if anything goes wrong).

```python
from fraudcrawler import Language, Location, Deepness, Prompt
# Setup the search
search_term = "sildenafil"
language = Language(name="German")
location = Location(name="Switzerland")
deepness = Deepness(num_results=50)
prompts = [
    Prompt(
        name="relevance",
        context="This organization is interested in medical products and drugs.",
        system_prompt=(
            "You are a helpful and intelligent assistant. Your task is to classify any given product "
            "as either relevant (1) or not relevant (0), strictly based on the context and product details provided by the user. "
            "You must consider all aspects of the given context and make a binary decision accordingly. "
            "If the product aligns with the user's needs, classify it as 1 (relevant); otherwise, classify it as 0 (not relevant). "
            "Respond only with the number 1 or 0."
        ),
        allowed_classes=[0, 1],
    )
]
```

(Optional) Add search term enrichement. This will find related search terms (in a given language) and search for these as well.
```python
from fraudcrawler import Enrichment
deepness.enrichment = Enrichment(
    additional_terms=5,
    additional_urls_per_term=10
)
```

(Optional) Add marketplaces where we explicitely want to look for (this will focus your search as the :site parameter for a google search)
```python
from fraudcrawler import Host
marketplaces = [
    Host(name="International", domains="zavamed.com,apomeds.com"),
    Host(name="National", domains="netdoktor.ch, nobelpharma.ch"),
]
```

(Optional) Exclude urls (where you don't want to find products)
```python
excluded_urls = [
    Host(name="Compendium", domains="compendium.ch"),
]
```

(Optional) Exclude previously collected urls (intends to save credits)
```python
previously_collected_urls = [
    https://pharmaciedelabateliere.ch/shop/sante/douleurs-inflammations/dafalgan-cpr-eff-500-mg-16-pce/,
    https://eiche.ch/product/schmerzmittel-52cd81d5d206a/dafalgan-brausetabletten-1336653,
]
```

And finally run the pipeline
```python
# Execute the pipeline
client.execute(
    search_term=search_term,
    language=language,
    location=location,
    deepness=deepness,
    prompts=prompts,
    # marketplaces=marketplaces,    # Uncomment this for using marketplaces
    # excluded_urls=excluded_urls   # Uncomment this for using excluded_urls
    # previously_collected_urls=previously_collected_urls    # Uncomment this for using previously_selected_urls
)
```
This creates a file with name pattern `<search_term>_<language.code>_<location.code>_<datetime[%Y%m%d%H%M%S]>.csv` inside the folder `data/results/`.

Once the pipeline terminated the results can be loaded and examined as follows:
```python
df = client.load_results()
print(df.head(n=10))
```

If the client has been used to run multiple pipelines, an overview of the available results (for a given instance of 
`FraudCrawlerClient`) can be obtained with
```python
client.print_available_results()
```

## Contributing
see `CONTRIBUTING.md`

### Async Setup
The following image provides a schematic representation of the package's async setup.
![Async Setup](https://github.com/open-veanu/fraudcrawler/raw/master/docs/assets/images/Fraudcrawler_Async_Setup.svg)
