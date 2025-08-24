from base64 import b64encode
from collections import defaultdict
import logging
from pydantic import BaseModel
from typing import Dict, Iterator, List

import httpx
from tenacity import RetryCallState

from fraudcrawler.settings import ENRICHMENT_DEFAULT_LIMIT
from fraudcrawler.base.base import Location, Language
from fraudcrawler.base.retry import get_async_retry


logger = logging.getLogger(__name__)


class Keyword(BaseModel):
    """Model for keyword details (e.g. `Keyword(text="sildenafil", volume=100)`)."""

    text: str
    volume: int


class Enricher:
    """A client to interact with the DataForSEO API for enhancing searches (producing alternative search_terms)."""

    _auth_encoding = "ascii"
    _base_endpoint = "https://api.dataforseo.com"
    _suggestions_endpoint = "/v3/dataforseo_labs/google/keyword_suggestions/live"
    _keywords_endpoint = "/v3/dataforseo_labs/google/related_keywords/live"

    def __init__(self, http_client: httpx.AsyncClient, user: str, pwd: str):
        """Initializes the DataForSeoApiClient with the given username and password.

        Args:
            http_client: An httpx.AsyncClient to use for the async requests.
            user: The username for DataForSEO API.
            pwd: The password for DataForSEO API.
        """
        self._http_client = http_client
        self._user = user
        self._pwd = pwd
        auth = f"{user}:{pwd}"
        auth = b64encode(auth.encode(self._auth_encoding)).decode(self._auth_encoding)
        self._headers = {
            "Authorization": f"Basic {auth}",
            "Content-Encoding": "gzip",
        }

    @staticmethod
    def _log_before(search_term: str, retry_state: RetryCallState | None) -> None:
        """Context aware logging before the request is made."""
        if retry_state:
            logger.debug(
                f'DataForSEO suggested search with search="{search_term}" (attempt {retry_state.attempt_number}).'
            )
        else:
            logger.debug(f"retry_state is {retry_state}, not logging before.")

    @staticmethod
    def _log_before_sleep(search_term: str, retry_state: RetryCallState | None) -> None:
        """Context aware logging before sleeping after a failed request."""
        if retry_state and retry_state.outcome:
            logger.warning(
                f'Attempt {retry_state.attempt_number} DataForSEO suggested search with search_term="{search_term}" '
                f"failed with error: {retry_state.outcome.exception()}. "
                f"Retrying in {retry_state.upcoming_sleep:.0f} seconds."
            )
        else:
            logger.debug(f"retry_state is {retry_state}, not logging before_sleep.")

    @staticmethod
    def _extract_items_from_data(data: dict) -> Iterator[dict]:
        """Extracts the items from the DataForSEO response.

        Args:
            data: The response data from DataForSEO.
        """
        tasks = (
            data.get("tasks") or []
        )  # in contrast to data.get("tasks", []) this handles the case where data["tasks"] is set to None
        for task in tasks:
            results = task.get("result") or []
            for result in results:
                items = result.get("items") or []
                yield from items

    @staticmethod
    def _parse_suggested_keyword(item: dict) -> Keyword:
        """Parses a keyword from an item in the DataForSEO suggested keyword search response.

        Args:
            item: An item from the DataForSEO response.
        """
        text = item["keyword"]
        volume = item["keyword_info"]["search_volume"]
        return Keyword(text=text, volume=volume)

    def _extract_suggested_keywords(self, data: dict) -> List[Keyword]:
        """Extracts the keywords from the DataForSEO response for suggested keywords.

        Args:
            data: The response data from DataForSEO.

        The DataForSEO results are of the form
        (c.f. https://docs.dataforseo.com/v3/dataforseo_labs/google/keyword_suggestions/live/?bash):
        {
          "tasks": [
            {
              "result": [
                {
                  "items": [
                    {
                      "keyword": <suggested-keyword>,
                      "keyword_info": {
                        "search_volume": <volume>
                      }
                    }
                  ]
                }
              ]
            }
          ]
        }

        Args:
            data: The response data from DataForSEO.
        """
        keywords = []
        for item in self._extract_items_from_data(data=data):
            try:
                keyword = self._parse_suggested_keyword(item)
                keywords.append(keyword)
            except Exception as e:
                logger.warning(f"Ignoring keyword due to error: {e}.")
        return keywords

    async def _get_suggested_keywords(
        self,
        search_term: str,
        language: Language,
        location: Location,
        limit: int = ENRICHMENT_DEFAULT_LIMIT,
    ) -> List[Keyword]:
        """Get keyword suggestions for a given search_term.

        Args:
            search_term: The search term to use for the query.
            language: The language to use for the search.
            location: The location to use for the search.
            limit: The upper limit of suggestions to get.
        """

        # Data must be a list of dictionaries, setting a number of search tasks; here we only have one task.
        data = [
            {
                "keyword": search_term,
                "language_name": language.name,
                "location_name": location.name,
                "limit": limit,
                "include_serp_info": True,
                "include_seed_keyword": True,
            }
        ]
        url = f"{self._base_endpoint}{self._suggestions_endpoint}"
        logger.debug(
            f'DataForSEO search suggested keywords with url="{url}" and data="{data}".'
        )

        # Perform the request and retry if necessary. There is some context aware logging
        #  - `before`: before the request is made (or before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            search_term=search_term, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            search_term=search_term, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                response = await self._http_client.post(
                    url=url, headers=self._headers, json=data
                )
                response.raise_for_status()

        # Extract the keywords from the response
        data_suggested_keywords = response.json()
        keywords = self._extract_suggested_keywords(data=data_suggested_keywords)

        logger.debug(f"Found {len(keywords)} suggestions from DataForSEO search.")
        return keywords

    @staticmethod
    def _parse_related_keyword(item: dict) -> Keyword:
        """Parses a keyword from an item in the DataForSEO related keyword search response.

        Args:
            item: An item from the DataForSEO response.
        """
        text = item["keyword_data"]["keyword"]
        volume = item["keyword_data"]["keyword_info"]["search_volume"]
        return Keyword(text=text, volume=volume)

    def _extract_related_keywords(self, data: dict) -> List[Keyword]:
        """Extracts the keywords from the DataForSEO response for related keywords.

        Args:
            data: The response data from DataForSEO.

        The DataForSEO results are of the form
        (c.f. https://docs.dataforseo.com/v3/dataforseo_labs/google/related_keywords/live/?bash):
        {
          "tasks": [
            {
              "result": [
                {
                  "items": [
                    {
                      "keyword_data": {
                        "keyword": <related-keyword>,
                        "keyword_info": {
                          "search_volume": <volume>
                        }
                      }
                    }
                  ]
                }
              ]
            }
          ]
        }

        Args:
            data: The response data from DataForSEO.
        """
        keywords = []
        for item in self._extract_items_from_data(data=data):
            try:
                keyword = self._parse_related_keyword(item)
                keywords.append(keyword)
            except Exception as e:
                logger.warning(f"Ignoring keyword due to error: {e}.")
        return keywords

    async def _get_related_keywords(
        self,
        search_term: str,
        language: Language,
        location: Location,
        limit: int = ENRICHMENT_DEFAULT_LIMIT,
    ) -> List[Keyword]:
        """Get related keywords for a given search_term.

        Args:
            search_term: The search term to use for the query.
            location: The location to use for the search.
            language: The language to use for the search.
            limit: The upper limit of suggestions to get.
        """

        # Data must be a list of dictionaries setting a number of search tasks; here we only have one task.
        data = [
            {
                "keyword": search_term,
                "language_name": language.name,
                "location_name": location.name,
                "limit": limit,
            }
        ]
        url = f"{self._base_endpoint}{self._keywords_endpoint}"
        logger.debug(
            f'DataForSEO search related keywords with url="{url}" and data="{data}".'
        )

        # Perform the request and retry if necessary. There is some context aware logging
        #  - `before`: before the request is made (or before retrying)
        #  - `before_sleep`: if the request fails before sleeping
        retry = get_async_retry()
        retry.before = lambda retry_state: self._log_before(
            search_term=search_term, retry_state=retry_state
        )
        retry.before_sleep = lambda retry_state: self._log_before_sleep(
            search_term=search_term, retry_state=retry_state
        )
        async for attempt in retry:
            with attempt:
                response = await self._http_client.post(
                    url=url, headers=self._headers, json=data
                )
                response.raise_for_status()

        # Extract the keywords from the response
        data_related_keywords = response.json()
        keywords = self._extract_related_keywords(data=data_related_keywords)

        logger.debug(f"Found {len(keywords)} related keywords from DataForSEO search.")
        return keywords

    async def enrich(
        self,
        search_term: str,
        language: Language,
        location: Location,
        n_terms: int,
    ) -> List[str]:
        """Applies the enrichment to a search_term.

        Args:
            search_term: The search term to use for the query.
            location: The location to use for the search.
            language: The language to use for the search.
            n_terms: The number of additional terms
        """
        logger.info(
            f'Applying enrichment for search_term="{search_term}" and n_terms="{n_terms}".'
        )
        # Get the additional suggested keywords
        try:
            suggested = await self._get_suggested_keywords(
                search_term=search_term,
                location=location,
                language=language,
                limit=n_terms,
            )
        except Exception as e:
            logger.error(
                f"Error fetching suggested keywords for search_term='{search_term}': {e}"
            )
            suggested = []

        # Get the additional related keywords
        try:
            related = await self._get_related_keywords(
                search_term=search_term,
                location=location,
                language=language,
                limit=n_terms,
            )
        except Exception as e:
            logger.error(
                f"Error fetching related keywords for search_term='{search_term}': {e}"
            )
            related = []

        # Remove original keyword and aggregate them by volume
        keywords = [kw for kw in suggested + related if kw.text != search_term]
        kw_vol: Dict[str, int] = defaultdict(int)
        for kw in keywords:
            kw_vol[kw.text] = max(kw.volume, kw_vol[kw.text])
        keywords = [Keyword(text=k, volume=v) for k, v in kw_vol.items()]
        logger.debug(f"Found {len(keywords)} additional unique keywords.")

        # Sort the keywords by volume and get the top n_terms
        keywords = sorted(keywords, key=lambda kw: kw.volume, reverse=True)
        terms = [kw.text for kw in keywords[:n_terms]]
        logger.info(f"Produced {len(terms)} additional search_terms.")
        return terms
