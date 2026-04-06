"""
AXIOM Paper Normaliser & Deduplicator
======================================
Cleans, normalises, and deduplicates Paper objects before they enter
the warehouse. This runs in the ingestion pipeline between raw fetch
and storage.

Pipeline:
    raw papers → normalise() → deduplicate() → quality enrichment → store

Usage:
    from axiom.normaliser import Normaliser

    norm = Normaliser()
    clean   = norm.normalise_many(raw_papers)
    unique  = norm.deduplicate(clean, existing_titles=existing)
"""

from __future__ import annotations

import re
import unicodedata
import hashlib
import logging
from typing import Optional

from .models import Paper
from .quality import enrich_quality

logger = logging.getLogger(__name__)

# Characters that are noise in titles / abstracts from different sources
_WHITESPACE_RE  = re.compile(r"\s+")
_HTML_TAG_RE    = re.compile(r"<[^>]+>")
_CONTROL_RE     = re.compile(r"[\x00-\x1f\x7f]")
_LATEX_CMD_RE   = re.compile(r"\\[a-zA-Z]+\{([^}]*)\}")   # \textbf{x} → x
_LATEX_SOLO_RE  = re.compile(r"\\[a-zA-Z]+")             # \cmd → ""
_LATEX_BRACE_RE = re.compile(r"[{}]")


def _clean_text(text: str | None) -> Optional[str]:
    """Normalise a free-text field: decode entities, strip HTML, collapse whitespace."""
    if not text:
        return None
    s = text
    s = _HTML_TAG_RE.sub(" ", s)
    s = _CONTROL_RE.sub(" ", s)
    s = _LATEX_CMD_RE.sub(r"\1", s)    # \textbf{word} → word
    s = _LATEX_SOLO_RE.sub(" ", s)     # remaining \cmd → space
    s = _LATEX_BRACE_RE.sub("", s)
    # Normalise unicode (e.g. accented chars → canonical form)
    s = unicodedata.normalize("NFC", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s or None


def _clean_title(title: str | None) -> str:
    """Clean and title-case a paper title."""
    cleaned = _clean_text(title)
    if not cleaned:
        return "Untitled"
    # Collapse any remaining double spaces
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _clean_authors(authors: list[str]) -> list[str]:
    """Deduplicate and clean an author list."""
    seen: set[str] = set()
    result: list[str] = []
    for a in authors:
        cleaned = _clean_text(a)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _clean_doi(doi: str | None) -> Optional[str]:
    """Strip URL prefix from DOIs and lowercase."""
    if not doi:
        return None
    doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "DOI:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi.strip().lower() or None


def _fingerprint(paper: Paper) -> str:
    """
    Stable fingerprint for deduplication.

    Priority:
      1. DOI (most reliable cross-source identifier)
      2. arXiv ID extracted from URL
      3. Normalised title + first author + year
    """
    if paper.doi:
        return f"doi:{paper.doi.lower()}"

    if paper.url:
        arxiv_match = re.search(r"arxiv\.org/abs/([\w.]+)", paper.url)
        if arxiv_match:
            return f"arxiv:{arxiv_match.group(1)}"

    if paper.source_id:
        return f"id:{paper.source_id.lower()}"

    # Title-based fingerprint
    norm_title = re.sub(r"[^\w]", "", paper.title.lower())
    first_author = re.sub(r"[^\w]", "", paper.authors[0].lower()) if paper.authors else ""
    year = str(paper.year) if paper.year else "unknown"
    raw = f"{norm_title}:{first_author}:{year}"
    return "hash:" + hashlib.sha1(raw.encode()).hexdigest()[:16]


class Normaliser:
    """
    Cleans and deduplicates Paper objects.

    Stateless — can be reused across ingestion runs.
    """

    def normalise(self, paper: Paper) -> Paper:
        """
        Return a cleaned copy of the paper.
        Does NOT modify the original.
        """
        data = paper.model_dump()

        data["title"]    = _clean_title(data.get("title"))
        data["abstract"] = _clean_text(data.get("abstract"))
        data["venue"]    = _clean_text(data.get("venue"))
        data["authors"]  = _clean_authors(data.get("authors") or [])
        data["doi"]      = _clean_doi(data.get("doi"))

        # Clamp relevance_score to valid range in case of bad upstream data
        rs = data.get("relevance_score", 0.0)
        data["relevance_score"] = max(0.0, min(1.0, float(rs) if rs else 0.0))

        # Truncate abstract to 2000 chars max (keeps embeddings consistent)
        if data["abstract"] and len(data["abstract"]) > 2000:
            data["abstract"] = data["abstract"][:2000].rsplit(" ", 1)[0] + "…"

        cleaned = Paper(**data)
        enrich_quality(cleaned)   # sets quality_rank from venue
        return cleaned

    def normalise_many(self, papers: list[Paper]) -> list[Paper]:
        """Normalise a batch of papers. Logs warnings for any that fail."""
        result: list[Paper] = []
        for p in papers:
            try:
                result.append(self.normalise(p))
            except Exception as e:
                logger.warning(f"Normaliser: skipped paper '{p.title[:50]}' — {e}")
        return result

    def deduplicate(
        self,
        papers: list[Paper],
        existing_fingerprints: set[str] | None = None,
    ) -> tuple[list[Paper], set[str]]:
        """
        Remove duplicate papers within the batch and against an existing set.

        Args:
            papers: Incoming papers (already normalised).
            existing_fingerprints: Fingerprints already in the warehouse.
                                   Pass None to deduplicate within batch only.

        Returns:
            (unique_papers, all_fingerprints)
              unique_papers      — papers not seen before
              all_fingerprints   — merged set (existing + new)
        """
        seen: set[str] = set(existing_fingerprints or [])
        unique: list[Paper] = []

        for p in papers:
            fp = _fingerprint(p)
            if fp not in seen:
                seen.add(fp)
                unique.append(p)

        dupes = len(papers) - len(unique)
        if dupes:
            logger.debug(f"Normaliser: removed {dupes} duplicate(s) from batch of {len(papers)}")

        return unique, seen

    def fingerprint(self, paper: Paper) -> str:
        """Public access to the fingerprint function."""
        return _fingerprint(paper)
