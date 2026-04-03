"""
API clients for academic paper sources.

Sources:
  - Semantic Scholar (primary — free, no key needed for basic use)
  - ArXiv (preprints, free)

Rate limiting:
  - Semantic Scholar allows ~1 req/sec without an API key.
    Get a free API key at https://www.semanticscholar.org/product/api
    and set SEMANTIC_SCHOLAR_API_KEY in your .env for higher limits.
  - ArXiv recommends no more than 1 req/3sec.
"""

from __future__ import annotations
import time
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
from .models import Paper

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "https://export.arxiv.org/api/query"

FIELDS = "title,authors,year,venue,externalIds,citationCount,abstract,openAccessPdf"


def _is_rate_limit(exc: BaseException) -> bool:
    """Return True only for 429 and 503 so we don't retry 4xx errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


class SemanticScholarClient:
    """
    Wrapper around the Semantic Scholar Graph API.

    Free tier: ~1 request/second.
    With API key: much higher limits.

    Get a free key at https://www.semanticscholar.org/product/api
    then add to your .env:
        SEMANTIC_SCHOLAR_API_KEY=your-key-here
    """

    def __init__(self, api_key: str | None = None):
        headers = {
            "Accept": "application/json",
            "User-Agent": "AXIOM-LiteratureAgent/0.1 (contact@voidstudio.tech)",
        }
        if api_key:
            headers["x-api-key"] = api_key
            self._has_key = True
        else:
            self._has_key = False

        self._client = httpx.Client(
            base_url=SEMANTIC_SCHOLAR_BASE,
            headers=headers,
            timeout=30.0,
        )
        # Polite delay between requests (seconds)
        self._min_delay = 1.1 if not self._has_key else 0.2
        self._last_request = 0.0

    def _wait(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_request = time.time()

    def _handle_429(self, response: httpx.Response):
        """Read Retry-After header and sleep accordingly."""
        retry_after = response.headers.get("Retry-After", "").strip()
        try:
            wait = max(int(retry_after), 30)
        except (ValueError, TypeError):
            wait = 30
        print(f"  [rate limit] Semantic Scholar asked us to wait {wait}s — pausing...")
        time.sleep(wait)

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=5, max=60),
    )
    def search(self, query: str, limit: int = 20, year_range: str | None = None) -> list[Paper]:
        params: dict = {
            "query": query,
            "limit": min(limit, 100),
            "fields": FIELDS,
        }
        if year_range:
            params["year"] = year_range

        for attempt in range(6):
            self._wait()
            resp = self._client.get("/paper/search", params=params)

            if resp.status_code == 429:
                if attempt < 5:
                    self._handle_429(resp)
                    continue
                resp.raise_for_status()

            if resp.status_code == 200:
                data = resp.json()
                return [self._parse(item) for item in data.get("data", [])]

            resp.raise_for_status()

        return []

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=30),
    )
    def get_paper(self, paper_id: str) -> Paper | None:
        for attempt in range(3):
            self._wait()
            try:
                resp = self._client.get(f"/paper/{paper_id}", params={"fields": FIELDS})
                if resp.status_code == 429:
                    if attempt < 2:
                        self._handle_429(resp)
                        continue
                resp.raise_for_status()
                return self._parse(resp.json())
            except httpx.HTTPStatusError:
                return None
        return None

    def _parse(self, item: dict) -> Paper:
        authors = [a.get("name", "") for a in item.get("authors", [])]
        ext = item.get("externalIds") or {}
        url = None
        if item.get("openAccessPdf"):
            url = item["openAccessPdf"].get("url")
        if not url and ext.get("ArXiv"):
            url = f"https://arxiv.org/abs/{ext['ArXiv']}"
        if not url:
            ss_id = item.get("paperId")
            if ss_id:
                url = f"https://www.semanticscholar.org/paper/{ss_id}"

        return Paper(
            title=item.get("title") or "Untitled",
            authors=authors,
            year=item.get("year"),
            venue=item.get("venue") or None,
            abstract=item.get("abstract") or None,
            citation_count=item.get("citationCount"),
            url=url,
            source="semantic_scholar",
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ArxivClient:
    """
    Wrapper around the ArXiv Atom API.
    Free, no key required. Recommended: no more than 1 req/3sec.
    """

    def __init__(self):
        self._client = httpx.Client(timeout=30.0)
        self._last_request = 0.0
        self._min_delay = 3.0  # ArXiv TOS: max 1 req/3sec

    def _wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)
        self._last_request = time.time()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=30),
    )
    def search(self, query: str, limit: int = 10) -> list[Paper]:
        import xml.etree.ElementTree as ET

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        self._wait()
        resp = self._client.get(ARXIV_BASE, params=params)
        resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        papers = []

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)
            id_el = entry.find("atom:id", ns)

            title = title_el.text.strip().replace("\n", " ") if title_el is not None else "Untitled"
            abstract = summary_el.text.strip() if summary_el is not None else None
            year = int(published_el.text[:4]) if published_el is not None else None
            url = id_el.text.strip() if id_el is not None else None

            authors = [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]

            papers.append(Paper(
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                url=url,
                venue="arXiv preprint",
                source="arxiv",
            ))

        return papers

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
