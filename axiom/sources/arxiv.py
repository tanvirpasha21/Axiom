"""
Async ArXiv source client.

Uses httpx async client + ArXiv Atom API.
Rate limit: 1 req / 3 sec per ArXiv TOS.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from ..models import Paper

logger = logging.getLogger(__name__)

ARXIV_BASE  = "https://export.arxiv.org/api/query"
_NS         = {"atom": "http://www.w3.org/2005/Atom"}
_MIN_DELAY  = 3.1   # seconds between requests (ArXiv TOS)


class AsyncArxivClient:
    """
    Async wrapper around the ArXiv Atom API.

    Usage:
        async with AsyncArxivClient() as client:
            papers = await client.search("fraud detection transformers", limit=20)
    """

    def __init__(self):
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
        """Enforce ArXiv's 1-req/3-sec rate limit."""
        import time
        elapsed = time.time() - self._last_request
        if elapsed < _MIN_DELAY:
            await asyncio.sleep(_MIN_DELAY - elapsed)
        import time
        self._last_request = time.time()

    async def search(
        self,
        query: str,
        limit: int = 20,
        year_range: Optional[str] = None,
    ) -> list[Paper]:
        """
        Search ArXiv and return Paper objects.

        Args:
            query:      Full-text search query.
            limit:      Max results to return (capped at 200).
            year_range: Optional "YYYY-YYYY" filter applied post-fetch
                        (ArXiv API doesn't support year filtering directly).
        """
        assert self._client, "Use as async context manager: async with AsyncArxivClient() as c:"

        limit = min(limit, 200)
        await self._wait()

        params = {
            "search_query": f"all:{query}",
            "start":        0,
            "max_results":  limit,
            "sortBy":       "relevance",
            "sortOrder":    "descending",
        }

        try:
            resp = await self._client.get(ARXIV_BASE, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"ArXiv fetch failed for '{query}': {e}")
            return []

        papers = self._parse(resp.text)

        if year_range:
            papers = _filter_year(papers, year_range)

        logger.info(f"ArXiv: '{query}' → {len(papers)} papers")
        return papers

    def _parse(self, xml_text: str) -> list[Paper]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"ArXiv XML parse error: {e}")
            return []

        papers: list[Paper] = []
        for entry in root.findall("atom:entry", _NS):
            try:
                papers.append(self._parse_entry(entry))
            except Exception as e:
                logger.debug(f"ArXiv: skipped entry — {e}")

        return papers

    def _parse_entry(self, entry) -> Paper:
        def txt(tag: str) -> Optional[str]:
            el = entry.find(tag, _NS)
            return el.text.strip() if el is not None and el.text else None

        title    = (txt("atom:title") or "Untitled").replace("\n", " ")
        abstract = txt("atom:summary")
        url      = txt("atom:id")
        year     = None
        doi      = None

        published = txt("atom:published")
        if published:
            year = int(published[:4])

        # Extract arXiv ID from URL
        source_id = None
        if url:
            m = re.search(r"abs/([\w.]+)", url)
            source_id = m.group(1) if m else None

        # DOI link if present
        for link in entry.findall("atom:link", _NS):
            if link.get("title") == "doi":
                doi = link.get("href", "").replace("http://dx.doi.org/", "")

        authors = [
            a.find("atom:name", _NS).text
            for a in entry.findall("atom:author", _NS)
            if a.find("atom:name", _NS) is not None
        ]

        # ArXiv categories → venue
        cats = [c.get("term", "") for c in entry.findall("atom:category", _NS)]
        venue = "arXiv preprint"
        if cats:
            venue = f"arXiv:{cats[0]}"

        return Paper(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            url=url,
            doi=doi,
            source_id=source_id,
            venue=venue,
            source="arxiv",
        )


def _filter_year(papers: list[Paper], year_range: str) -> list[Paper]:
    parts = year_range.split("-")
    if len(parts) != 2:
        return papers
    y_min = int(parts[0]) if parts[0].strip() else None
    y_max = int(parts[1]) if parts[1].strip() else None

    filtered = []
    for p in papers:
        if y_min and p.year and p.year < y_min:
            continue
        if y_max and p.year and p.year > y_max:
            continue
        filtered.append(p)
    return filtered
