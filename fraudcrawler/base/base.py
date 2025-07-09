import json
import logging
from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings
import re
from typing import List

import aiohttp

from fraudcrawler.settings import (
    GOOGLE_LANGUAGES_FILENAME,
    GOOGLE_LOCATIONS_FILENAME,
)

logger = logging.getLogger(__name__)

# Load google locations and languages
with open(GOOGLE_LOCATIONS_FILENAME, "r") as gfile:
    _locs = json.load(gfile)
_LOCATION_CODES = {loc["name"]: loc["country_code"].lower() for loc in _locs}
with open(GOOGLE_LANGUAGES_FILENAME, "r") as gfile:
    _langs = json.load(gfile)
_LANGUAGE_CODES = {lang["language_name"]: lang["language_code"] for lang in _langs}


# Base classes
class Setup(BaseSettings):
    """Class for loading environment variables."""

    # Crawler ENV variables
    serpapi_key: str
    dataforseo_user: str
    dataforseo_pwd: str
    zyteapi_key: str
    openaiapi_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Host(BaseModel):
    """Model for host details (e.g. `Host(name="Galaxus", domains="galaxus.ch, digitec.ch")`)."""

    name: str
    domains: str | List[str]

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        """Make it lowercase and strip 'www.' and 'https?://' prefixes from the domain."""
        domain = domain.strip().lower()
        return re.sub(r"^(https?://)?(www\.)?", "", domain)

    @field_validator("domains", mode="before")
    def normalize_domains(cls, val):
        if isinstance(val, str):
            val = val.split(",")
        return [cls._normalize_domain(dom.strip()) for dom in val]


class Location(BaseModel):
    """Model for location details (e.g. `Location(name="Switzerland", code="ch")`)."""

    name: str
    code: str = ""

    @model_validator(mode="before")
    def set_code(cls, values):
        """Set the location code if not provided and make it lower case."""
        name = values.get("name")
        code = values.get("code")
        if code is None or not len(code):
            code = _LOCATION_CODES.get(name)
            if code is None:
                raise ValueError(f'Location code not found for location name="{name}"')
        code = code.lower()
        return {"name": name, "code": code}


class Language(BaseModel):
    """Model for language details (e.g. `Language(name="German", code="de")`)."""

    name: str
    code: str = ""

    @model_validator(mode="before")
    def set_code(cls, values):
        """Set the language code if not provided and make it lower case."""
        name = values.get("name")
        code = values.get("code")
        if code is None or not len(code):
            code = _LANGUAGE_CODES.get(name)
            if code is None:
                raise ValueError(f'Language code not found for language name="{name}"')
        code = code.lower()
        return {"name": name, "code": code}


class Enrichment(BaseModel):
    """Model for enriching initial search_term with alternative ones."""

    additional_terms: int
    additional_urls_per_term: int


class Deepness(BaseModel):
    """Model for search depth."""

    num_results: int
    enrichment: Enrichment | None = None


class Prompt(BaseModel):
    """Model for prompts."""

    name: str
    context: str
    system_prompt: str
    allowed_classes: List[int]

    @field_validator("allowed_classes", mode="before")
    def check_for_positive_value(cls, val):
        """Check if all values are positive."""
        if not all(isinstance(i, int) and i >= 0 for i in val):
            raise ValueError("all values in allowed_classes must be positive integers.")
        return val


class AsyncClient:
    """Base class for sub-classes using async HTTP requests."""

    @staticmethod
    async def get(
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Async GET request of a given URL returning the data."""
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url=url, params=params) as response:
                response.raise_for_status()
                json_ = await response.json()
        return json_

    @staticmethod
    async def post(
        url: str,
        headers: dict | None = None,
        data: List[dict] | dict | None = None,
        auth: aiohttp.BasicAuth | None = None,
    ) -> dict:
        """Async POST request of a given URL returning the data."""
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url=url, json=data, auth=auth) as response:
                response.raise_for_status()
                json_ = await response.json()
        return json_
