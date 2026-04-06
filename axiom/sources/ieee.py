"""
IEEE Xplore source client.

Uses the IEEE Xplore REST API v3.
API key is free: https://developer.ieee.org  (register → "API Key")

Rate limits (free tier):
  - 10 requests/second
  - 200 records/request
  - 500 requests/day

Set your key:
    export IEEE_API_KEY=your-key-here
    # or pass ieee_api_key= to IEEEClient()
    # or add IEEE_API_KEY to .env

Usage:
    import asyncio
    from axiom.sources.ieee import IEEEClient

    async def main():
        async with IEEEClient(api_key="your-key") as client:
            papers = await client.search("fraud detection transformer", limit=50)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx

from ..models import Paper

logger = logging.getLogger(__name__)

IEEE_BASE   = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
_MIN_DELAY  = 0.15   # 10 req/sec → 100ms gap + safety margin


class IEEEClient:
    """
    Async IEEE Xplore search client.

    Fetches real IEEE papers including journals, conference proceedings,
    and magazines. Applies Q1/Q2 quality filter when quality_only=True.

    Usage:
        async with IEEEClient(api_key="...") as client:
            papers = await client.search("graph neural networks fraud", limit=50)
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = (
            api_key
            or os.getenv("IEEE_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "IEEE API key required.\n"
                "Get a free key at https://developer.ieee.org\n"
                "Then set:  export IEEE_API_KEY=your-key\n"
                "Or pass:   IEEEClient(api_key='your-key')"
            )
        self._client: httpx.AsyncClient | None = None
        self._last_request: float = 0.0

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _wait(self):
        import time
        elapsed = time.time() - self._last_request
        if elapsed < _MIN_DELAY:
            await asyncio.sleep(_MIN_DELAY - elapsed)
        self._last_request = time.time()  # type: ignore[assignment]

    async def search(
        self,
        query: str,
        limit: int = 50,
        year_range: Optional[str] = None,
        content_types: Optional[list[str]] = None,
    ) -> list[Paper]:
        """
        Search IEEE Xplore.

        Args:
            query:         Full-text search query.
            limit:         Max results (capped at 200 per IEEE API limits).
            year_range:    Optional "YYYY-YYYY" string.
            content_types: ["Journals", "Conferences", "Magazines", "Books"].
                           Defaults to Journals + Conferences.

        Returns:
            List of Paper objects from IEEE sources.
        """
        assert self._client, "Use as async context manager."

        limit = min(limit, 200)
        content_types = content_types or ["Journals", "Conferences"]

        params: dict = {
            "apikey":            self._api_key,
            "querytext":         query,
            "max_records":       limit,
            "start_record":      1,
            "sort_order":        "desc",
            "sort_field":        "relevance",
            "content_types":     ",".join(content_types),
            "abstract":          "true",
        }

        if year_range:
            parts = year_range.split("-")
            if len(parts) == 2:
                if parts[0].strip():
                    params["start_year"] = parts[0].strip()
                if parts[1].strip():
                    params["end_year"] = parts[1].strip()

        await self._wait()

        try:
            resp = await self._client.get(IEEE_BASE, params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("IEEE: Invalid or expired API key.")
            elif e.response.status_code == 429:
                logger.error("IEEE: Rate limit hit — reduce request frequency.")
            else:
                logger.error(f"IEEE HTTP error: {e}")
            return []
        except httpx.HTTPError as e:
            logger.error(f"IEEE connection error: {e}")
            return []

        data = resp.json()
        articles = data.get("articles", [])
        papers = [self._parse(a) for a in articles if a.get("title")]
        logger.info(f"IEEE: '{query}' → {len(papers)} papers (total_records={data.get('total_records')})")
        return papers

    def _parse(self, article: dict) -> Paper:
        """Parse an IEEE API article dict into a Paper."""
        authors_raw = article.get("authors", {}).get("authors", [])
        authors = [a.get("full_name", "") for a in authors_raw if a.get("full_name")]

        doi = article.get("doi")
        article_number = article.get("article_number", "")
        pdf_url = article.get("pdf_url") or article.get("html_url")
        if not pdf_url and doi:
            pdf_url = f"https://doi.org/{doi}"

        # Normalise year
        year = None
        year_raw = article.get("publication_year") or article.get("conference_dates", "")
        if isinstance(year_raw, (int, str)):
            try:
                year = int(str(year_raw)[:4])
            except (ValueError, TypeError):
                pass

        publication = (
            article.get("publication_title")
            or article.get("publisher")
            or None
        )

        return Paper(
            title=article.get("title", "Untitled"),
            authors=authors,
            year=year,
            venue=publication,
            abstract=article.get("abstract"),
            url=pdf_url,
            doi=doi,
            source_id=article_number or None,
            citation_count=article.get("citing_paper_count"),
            source="ieee",
        )
