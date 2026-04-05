"""
Paper retrieval sources for AXIOM.

Three sources — no external academic APIs, no rate limits, no keys required:

  LocalPaperStore  — Your own trainable paper knowledge base (JSONL on disk).
                     Add papers from ArXiv, JSON exports, or manual entry.
                     This is your primary training corpus.

  LLMPaperClient   — Uses the LLM's built-in training knowledge to recall papers.
                     No internet required with a local Ollama backend.

  ArxivClient      — Live ArXiv preprint search. Free, no key needed.
                     Respects 1 req/3sec per ArXiv TOS.
"""

from __future__ import annotations
import json
import time
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .models import Paper

logger = logging.getLogger(__name__)

ARXIV_BASE = "https://export.arxiv.org/api/query"


# ---------------------------------------------------------------------------
# Local trainable store
# ---------------------------------------------------------------------------

class LocalPaperStore:
    """
    A local, trainable paper knowledge base persisted as a JSONL file.

    Think of this as your private academic library. Train it by adding papers,
    then point LiteratureAgent at it with paper_source="local" (or "local+llm",
    "local+arxiv", "local+llm+arxiv").

    Adding papers (training):
        store = LocalPaperStore("my_papers.jsonl")

        # Import from ArXiv
        from axiom.api_clients import ArxivClient
        arxiv = ArxivClient()
        store.add_many(arxiv.search("fraud detection transformers", limit=20))

        # Import from a JSON/JSONL file (Zotero export, another AXIOM store, etc.)
        store.import_from_file("exported_papers.json")

        # Add a single paper manually
        from axiom.models import Paper
        store.add(Paper(
            title="My Paper Title",
            authors=["Author Name"],
            year=2024,
            venue="NeurIPS",
            abstract="Abstract text here...",
        ))

        print(store.stats())

    Searching:
        papers = store.search("fraud detection", limit=20, year_range="2020-2024")
    """

    def __init__(self, db_path: str = "axiom_papers.jsonl"):
        self._path = Path(db_path)
        self._papers: list[Paper] = []
        self._load()

    # ------------------------------------------------------------------
    # Training / adding papers
    # ------------------------------------------------------------------

    def add(self, paper: Paper) -> bool:
        """Add a single paper. Returns True if added, False if duplicate."""
        existing = {p.title.lower() for p in self._papers}
        if paper.title.lower() in existing:
            return False
        self._papers.append(paper)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(paper.model_dump_json() + "\n")
        return True

    def add_many(self, papers: list[Paper]) -> int:
        """
        Add multiple papers, skipping duplicates.
        Returns the count of newly added papers.
        """
        existing = {p.title.lower() for p in self._papers}
        added = 0
        with self._path.open("a", encoding="utf-8") as f:
            for paper in papers:
                if paper.title.lower() not in existing:
                    self._papers.append(paper)
                    existing.add(paper.title.lower())
                    f.write(paper.model_dump_json() + "\n")
                    added += 1
        return added

    def import_from_file(self, path: str) -> int:
        """
        Import papers from a JSON or JSONL file.

        Accepts:
          - A JSON array of paper objects  (e.g. from Zotero export)
          - A JSONL file with one paper per line  (another AXIOM store)
          - Any file with Paper-compatible dicts

        Returns the number of newly added papers.
        """
        content = Path(path).read_text(encoding="utf-8").strip()
        items: list[dict] = []

        # Try JSON array first
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                items = parsed
        except json.JSONDecodeError:
            # Fall back to JSONL
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        papers = []
        for item in items:
            if isinstance(item, dict) and item.get("title"):
                try:
                    papers.append(Paper(**item))
                except Exception:
                    pass

        return self.add_many(papers)

    def remove(self, title: str) -> bool:
        """Remove a paper by exact title. Rewrites the store file. Returns True if found."""
        before = len(self._papers)
        self._papers = [p for p in self._papers if p.title.lower() != title.lower()]
        if len(self._papers) == before:
            return False
        self._rewrite()
        return True

    def clear(self) -> int:
        """Remove all papers from the store. Returns count removed."""
        count = len(self._papers)
        self._papers = []
        if self._path.exists():
            self._path.unlink()
        return count

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20, year_range: str | None = None) -> list[Paper]:
        """
        Keyword search over the local store.

        Scoring: each query term found in the text scores 1 point;
        terms found in the title score an extra 2 points (title is weighted heavier).
        Results are sorted by score descending.
        """
        if not self._papers:
            logger.warning(
                "LocalPaperStore is empty. Add papers first:\n"
                "  store = LocalPaperStore()\n"
                "  store.add_many(ArxivClient().search('your topic', limit=30))\n"
                "Or import from a file: store.import_from_file('papers.json')"
            )
            return []

        terms = [t for t in query.lower().split() if len(t) > 2]
        year_min, year_max = _parse_year_range(year_range)

        scored: list[tuple[float, Paper]] = []
        for paper in self._papers:
            if year_min and paper.year and paper.year < year_min:
                continue
            if year_max and paper.year and paper.year > year_max:
                continue

            text = " ".join([
                paper.title,
                paper.abstract or "",
                " ".join(paper.authors),
                paper.venue or "",
            ]).lower()

            body_hits = sum(1 for t in terms if t in text)
            title_hits = sum(1 for t in terms if t in paper.title.lower())
            score = body_hits + title_hits * 2

            if score > 0:
                scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    def all_papers(self) -> list[Paper]:
        """Return every paper in the store."""
        return list(self._papers)

    def stats(self) -> dict:
        """Return summary statistics about the store."""
        years = [p.year for p in self._papers if p.year]
        sources: dict[str, int] = {}
        for p in self._papers:
            sources[p.source] = sources.get(p.source, 0) + 1
        return {
            "total": len(self._papers),
            "db_path": str(self._path.resolve()),
            "year_range": f"{min(years)}–{max(years)}" if years else "no years",
            "sources": sources,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._papers.append(Paper(**json.loads(line)))
                except Exception:
                    pass

    def _rewrite(self):
        with self._path.open("w", encoding="utf-8") as f:
            for paper in self._papers:
                f.write(paper.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# LLM knowledge retrieval
# ---------------------------------------------------------------------------

class LLMPaperClient:
    """
    Retrieves papers using the LLM's own training knowledge — zero external API calls.

    Works with any configured backend: Ollama (local/offline), Anthropic, OpenAI,
    or OpenRouter. The LLM is prompted to recall real papers it is confident exist.

    Trade-offs:
      + No API key, no rate limits, works fully offline with Ollama
      - Knowledge cutoff: papers after the model's training date won't appear
      - Smaller models may miss niche papers
      - Always cross-check important claims — LLMs can occasionally misremember details

    Use via paper_source="llm" or any compound source in LiteratureAgent.
    """

    _SYSTEM = (
        "You are an academic literature assistant with broad knowledge of published research. "
        "When given a query, return a JSON array of real papers you are confident exist. "
        "CRITICAL: Never invent titles, authors, venues, or any other detail. "
        "If you are uncertain about a specific field, omit it rather than guess. "
        "Respond with ONLY a valid JSON array — no explanation, no markdown fences."
    )

    def __init__(self, backend):
        # backend: any object with a .query(system, prompt, max_tokens) -> str method
        self._llm = backend

    def search(self, query: str, limit: int = 15, year_range: str | None = None) -> list[Paper]:
        year_clause = (
            f"\nOnly include papers published in the year range {year_range}."
            if year_range else ""
        )
        prompt = (
            f'Research query: "{query}"{year_clause}\n\n'
            f"Return a JSON array of up to {limit} real academic papers relevant to this query.\n"
            "Each element must follow this schema exactly:\n"
            '{\n'
            '  "title": "Exact paper title",\n'
            '  "authors": ["First Author", "Second Author"],\n'
            '  "year": 2023,\n'
            '  "venue": "Conference or Journal Name",\n'
            '  "abstract": "2-3 sentence description of the paper\'s contribution and findings",\n'
            '  "url": "https://arxiv.org/abs/... or DOI link if known, otherwise null",\n'
            '  "citation_count": estimated citations as integer or null\n'
            '}\n\n'
            "Return ONLY a valid JSON array. No explanation, no markdown."
        )
        raw = self._llm.query(self._SYSTEM, prompt, max_tokens=3000)
        return self._parse(raw)

    def _parse(self, raw: str) -> list[Paper]:
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            items = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            logger.warning("LLMPaperClient: failed to parse JSON response")
            return []
        if not isinstance(items, list):
            return []
        papers = []
        for item in items:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            papers.append(Paper(
                title=item.get("title", "Untitled"),
                authors=item.get("authors") or [],
                year=item.get("year"),
                venue=item.get("venue"),
                abstract=item.get("abstract"),
                url=item.get("url"),
                citation_count=item.get("citation_count"),
                source="llm_knowledge",
            ))
        return papers

    def close(self):
        pass  # backend is owned by LiteratureAgent


# ---------------------------------------------------------------------------
# ArXiv live search
# ---------------------------------------------------------------------------

class ArxivClient:
    """
    Live ArXiv preprint search via the public Atom API.
    Free, no key required. Enforces 1 req/3sec per ArXiv TOS.
    """

    def __init__(self):
        self._client = httpx.Client(timeout=30.0)
        self._last_request = 0.0
        self._min_delay = 3.0

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_year_range(year_range: str | None) -> tuple[int | None, int | None]:
    if not year_range:
        return None, None
    parts = year_range.split("-")
    if len(parts) != 2:
        return None, None
    y_min = int(parts[0]) if parts[0].strip() else None
    y_max = int(parts[1]) if parts[1].strip() else None
    return y_min, y_max
