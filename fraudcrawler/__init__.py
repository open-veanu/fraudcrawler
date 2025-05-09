from fraudcrawler.scraping.serp import SerpApi
from fraudcrawler.scraping.enrich import Enricher
from fraudcrawler.scraping.zyte import ZyteApi
from fraudcrawler.processing.processor import Processor
from fraudcrawler.base.orchestrator import Orchestrator, ProductItem
from fraudcrawler.base.client import FraudCrawlerClient
from fraudcrawler.base.base import (
    Deepness,
    Enrichment,
    Host,
    Language,
    DSsettings,
    Location,
    Prompt,
)
import logging

#logging.basicConfig(level=LOG_LVL.upper(), format=LOG_FMT, datefmt=LOG_DATE_FMT)
#logger = logging.getLogger(__name__)

# Avoid noisy logs from hpack, httpcore, urllib3, and openai (make it at least logger.INFO)
#level = max(getattr(logging, LOG_LVL), 20)
#logging.getLogger("hpack").setLevel(level=level)
#logging.getLogger("httpcore").setLevel(level=level)
#logging.getLogger("urllib3").setLevel(level=level)
#logging.getLogger("openai").setLevel(level=level)
#logger = logging.getLogger(__name__)

__all__ = [
    "SerpApi",
    "Enricher",
    "ZyteApi",
    "Processor",
    "Orchestrator",
    "ProductItem",
    "FraudCrawlerClient",
    "Language",
    "DSsettings",
    "Location",
    "Host",
    "Deepness",
    "Enrichment",
    "Prompt",
]
