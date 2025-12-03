from pathlib import Path
from typing import List

# Generic settings
ROOT_DIR = Path(__file__).parents[1]

# Service retry settings
# With the following setup (neglecting the jitter) we have 6 attempts with delays:
#   0s, 1s, 4s, 16s, 64s, 64s (because of the max delay)
RETRY_STOP_AFTER_ATTEMPT = 6
RETRY_INITIAL_DELAY = 1
RETRY_MAX_DELAY = 64
RETRY_EXP_BASE = 4
RETRY_JITTER = 1
RETRY_SKIP_IF_CODE = [400, 401, 403]  # Skip retrying on these HTTP status codes

# Search settings
GOOGLE_LOCATIONS_FILENAME = ROOT_DIR / "fraudcrawler" / "base" / "google-locations.json"
GOOGLE_LANGUAGES_FILENAME = ROOT_DIR / "fraudcrawler" / "base" / "google-languages.json"
SEARCH_DEFAULT_COUNTRY_CODES: List[str] = [
    # ".com",
]
TOPPREISE_SEARCH_PATHS = {
    "de": "produktsuche",
    "fr": "chercher",
    "default": "browse",
}
TOPPREISE_COMPARISON_PATHS = [
    "preisvergleich",
    "comparison-prix",
    "price-comparison",
]

# URL De-duplication settings
KNOWN_TRACKERS = [
    "srsltid",  # Search result click ID (used by some search engines)
    "utm_source",  # UTM: Source of the traffic (e.g., Google, Newsletter)
    "utm_medium",  # UTM: Medium such as CPC, email, social
    "utm_campaign",  # UTM: Campaign name (e.g., summer_sale)
    "utm_term",  # UTM: Keyword term (used in paid search)
    "utm_content",  # UTM: Used to differentiate similar links or ads
    "ar",  # Often used for ad region or targeting info
    "ps",  # Could refer to promotion source or partner segment
    "gclid",  # Google Ads click ID (auto-tagging)
    "gclsrc",  # Source of the GCLID (e.g., ads, search)
    "sku",  # Product SKU identifier, often used in ecommerce links
    "ref",  # Referrer username or source (e.g., GitHub ref links)
    "referral",  # Alternate form of referrer, often human-readable
    "aff_id",  # Affiliate identifier (ID-based)
    "aff",  # Short form for affiliate tag
    "affiliate",  # Affiliate tracking parameter (human-readable)
    "partner",  # Indicates marketing or distribution partner
    "fbclid",  # Facebook Click Identifier
    "msclkid",  # Microsoft/Bing Ads click identifier
    "twclid",  # Twitter Ads click identifier
    "variant",  # A/B test variant (used to test versions of pages)
    "session_id",  # Session tracking ID, should not persist across URLs
    "track",  # Generic flag used to enable/disable tracking
    "cid",  # Campaign ID (used in ads or emails)
    "campaignid",  # Alternate or long-form campaign ID
    "adgroup",  # Ad group identifier for campaigns
    "bannerid",  # Specific banner ad ID (for display ad tracking)
    "token",  # Often used to identify users or temporary sessions
    "tag",  # Affiliate or marketing tag (used for tracking)
    "hash",  # Generic hash identifier, often for state or cache
    "user",  # User ID or identifier passed in URL (should be avoided)
    "src",  # Generic source indicator, less formal than `utm_source`
    "selsort",  # Sorting parameter for search results
    "shid",  # Shop ID (used in ecommerce)
    "shoparea",  # Shop area (used in ecommerce)
    "shopid",  # Shop ID (used in ecommerce)
    "shoparea",  # Shop area (used in ecommerce)
]

# Enrichment settings
ENRICHMENT_DEFAULT_LIMIT = 10

# Zyte settings
ZYTE_DEFALUT_PROBABILITY_THRESHOLD = 0.1

# Exact match settings
EXACT_MATCH_PRODUCT_FIELDS = {
    "url_resolvedproduct_name",
    "product_description",
    "html",
}
EXACT_MATCH_FIELD_SEPARATOR = "\n"

# Async workers settings
DEFAULT_N_SRCH_WKRS = 2
DEFAULT_N_CNTX_WKRS = 23
DEFAULT_N_PROC_WKRS = 5

# HTTPX client settings
DEFAULT_HTTPX_TIMEOUT = {
    "timeout": 600,
    "connect": 5.0,
}
DEFAULT_HTTPX_LIMITS = {
    "max_connections": 1000,
    "max_keepalive_connections": 100,
}
DEFAULT_HTTPX_REDIRECTS = True
