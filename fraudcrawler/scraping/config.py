from pydantic import BaseModel, Field
from typing import List

from fraudcrawler import (
    SearchEngineName,
    Language,
    Location,
    Deepness,
    Host,
)

class ScrapingConfig(BaseModel):
    """Sets up the scraping pipeline step."""

    search_term: str = Field(description="The search term for the query.")
    search_engines: List[SearchEngineName] = Field(description="The list of search engines to use for the search query.")
    language: Language = Field(description="The language to use for the query.")
    location: Location = Field(description="The location to use for the query.")
    deepness: Deepness = Field(description="The search depth and enrichment details.")
    marketplaces: List[Host] | None = Field(default=None, description="The marketplaces to include in the search.")
    excluded_urls: List[Host] | None = Field(default=None, description="The URLs to exclude from the search.")
    previously_collected_urls: List[str] | None = Field(default=None, description="The URLs that have been collected previously and are ignored.")
