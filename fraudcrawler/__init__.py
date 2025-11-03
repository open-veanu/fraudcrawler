from fraudcrawler.scraping.search import Searcher, SearchEngineName
from fraudcrawler.scraping.enrich import Enricher
from fraudcrawler.scraping.url import URLCollector
from fraudcrawler.scraping.zyte import ZyteAPI
from fraudcrawler.scraping.config import ScrapingConfig
from fraudcrawler.processing.processor import Processor
from fraudcrawler.processing.config import ProcessingConfig
from fraudcrawler.base.orchestrator import Orchestrator
from fraudcrawler.base.client import FraudCrawlerClient
from fraudcrawler.base.base import (
    Deepness,
    Enrichment,
    Host,
    Language,
    DSsettings,
    Location,
    Prompt,
    ProductItem,
    HttpxAsyncClient,
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
    "Searcher",
    "SearchEngineName",
    "Enricher",
    "URLCollector",
    "ZyteAPI",
    "ScrapingConfig",
    "Processor",
    "ProcessingConfig",
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
    "HttpxAsyncClient",
]
