from pathlib import Path

# Generic settings
MAX_RETRIES = 3
RETRY_DELAY = 2
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR  = ROOT_DIR / "data" / "results"

# Serp settings
GOOGLE_LOCATIONS_FILENAME = ROOT_DIR / "fraudcrawler" / "base" / "google-locations.json"
GOOGLE_LANGUAGES_FILENAME = ROOT_DIR / "fraudcrawler" / "base" / "google-languages.json"

# Enrichment settings
ENRICHMENT_DEFAULT_LIMIT = 10

# Zyte settings
ZYTE_DEFALUT_PROBABILITY_THRESHOLD = 0.1

# Processor settings
PROCESSOR_DEFAULT_MODEL = "gpt-4o"
PROCESSOR_DEFAULT_IF_MISSING = -1
PROCESSOR_USER_PROMPT_TEMPLATE = (
    "Context: {context}\n\nProduct Details: {name}\n{description}\\n\nRelevance:"
)

# Orchestrator settings
PRODUCT_ITEM_DEFAULT_IS_RELEVANT = -1

# Async settings
DEFAULT_N_SERP_WKRS = 10
DEFAULT_N_ZYTE_WKRS = 10
DEFAULT_N_PROC_WKRS = 10
