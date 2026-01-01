"""
Semantic Scholar API Client

Provides a clean interface for querying the Semantic Scholar API.
"""

import requests
import json
import time
from typing import List, Dict, Optional

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import SEMANTIC_SCHOLAR_API_URL, SEMANTIC_SCHOLAR_API_KEY


class SemanticScholarClient:
    """
    Client for interacting with the Semantic Scholar API.

    Usage:
        client = SemanticScholarClient()
        papers = client.search_papers("topic modeling", limit=100)
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the client.

        Args:
            api_key: Optional API key. If not provided, uses environment variable.
        """
        self.api_key = api_key or SEMANTIC_SCHOLAR_API_KEY
        self.base_url = SEMANTIC_SCHOLAR_API_URL
        self.headers = {}

        if self.api_key:
            self.headers["x-api-key"] = self.api_key
            print("Using API key for authentication")
        else:
            print("No API key provided, using anonymous access")

    def search_papers(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0,
        fields: List[str] = None,
        fields_of_study: str = None,
    ) -> List[Dict]:
        """
        Search for papers matching a query.

        Args:
            query: Search query string
            limit: Maximum number of papers to return
            offset: Starting offset for pagination
            fields: List of fields to return (default: corpusId, title, abstract)
            fields_of_study: Filter by field of study (comma-separated)

        Returns:
            List of paper dictionaries
        """
        print(f"Fetching from Semantic Scholar API{limit} papers about '{query}'...")
        if fields_of_study:
            print(f"Field of study filter: {fields_of_study}")

        all_papers = []
        batch_size = 100  # API limit
        total_batches = (limit + batch_size - 1) // batch_size
        actual_total = None

        default_fields = fields or ["corpusId", "title", "abstract"]

        for batch in range(total_batches):
            current_offset = offset + batch * batch_size
            current_limit = min(batch_size, limit - batch * batch_size)

            # Stop if we've exceeded actual total
            if actual_total is not None and current_offset >= actual_total:
                print(f"Reached actual total papers ({actual_total}); stopping fetch")
                break

            print(
                f"Fetching batch {batch + 1}/{total_batches} papers (offset: {current_offset}, limit: {current_limit})..."
            )

            papers, total = self._fetch_batch(
                query, current_limit, current_offset, default_fields, fields_of_study
            )

            if batch == 0:
                print("API call successful!")
                print(f"Total papers: {total:,}")
                actual_total = total

            all_papers.extend(papers)
            print(f"Batch {batch + 1}: fetched {len(papers)} papers")

            # Rate limiting
            if batch < total_batches - 1:
                time.sleep(1)

        print(f"Total fetched {len(all_papers)}  papers")
        return all_papers

    def _fetch_batch(
        self,
        query: str,
        limit: int,
        offset: int,
        fields: List[str],
        fields_of_study: str = None,
        max_retries: int = 10,
    ) -> tuple:
        """Fetch a single batch of papers with retry logic."""
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "fields": ",".join(fields),
            "limit": limit,
            "offset": offset,
        }

        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study.strip()

        for retry in range(max_retries):
            try:
                response = requests.get(
                    url, params=params, headers=self.headers, timeout=30
                )

                if response.status_code == 429:
                    if retry < max_retries - 1:
                        print(
                            f"  Rate limited (429); waiting 5s before retry... (attempt {retry + 1})"
                        )
                        time.sleep(5)
                        continue
                    else:
                        print("  Reached max retries")
                        return [], 0

                response.raise_for_status()
                data = response.json()

                return data.get("data", []), data.get("total", 0)

            except requests.exceptions.RequestException as e:
                if retry < max_retries - 1:
                    print(
                        f"  API request failed: {e}; waiting 5s before retry... (attempt {retry + 1})"
                    )
                    time.sleep(5)
                else:
                    print(f"  Reached max retries: {e}")
                    return [], 0
            except json.JSONDecodeError as e:
                if retry < max_retries - 1:
                    print(f"  JSON parse failed: {e}; waiting 5s before retry...")
                    time.sleep(5)
                else:
                    print(f"  Reached max retries: {e}")
                    return [], 0

        return [], 0

    def get_paper_by_id(
        self, paper_id: str, fields: List[str] = None
    ) -> Optional[Dict]:
        """
        Get a paper by its ID.

        Args:
            paper_id: Paper ID (corpus ID, DOI, etc.)
            fields: Fields to return

        Returns:
            Paper dictionary or None
        """
        url = f"{self.base_url}/paper/{paper_id}"
        params = {}

        if fields:
            params["fields"] = ",".join(fields)

        try:
            response = requests.get(
                url, params=params, headers=self.headers, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to fetch paper: {e}")
            return None
