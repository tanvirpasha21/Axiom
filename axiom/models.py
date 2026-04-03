"""
Core Pydantic models for AXIOM.
All structured data returned by the agent is typed here.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A single academic paper with metadata and agent-computed signals."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    conflict_flag: bool = False
    supports_hypothesis: bool = False
    is_negative_result: bool = False
    source: str = "semantic_scholar"

    def __str__(self) -> str:
        authors_str = ", ".join(self.authors[:2])
        if len(self.authors) > 2:
            authors_str += " et al."
        year_str = f" ({self.year})" if self.year else ""
        venue_str = f" · {self.venue}" if self.venue else ""
        return f"{self.title} — {authors_str}{year_str}{venue_str}"


class ConflictPair(BaseModel):
    """Two papers that make opposing claims."""

    paper_a: str
    paper_b: str
    claim_a: str
    claim_b: str
    contested_topic: str
    severity: str = Field(default="moderate", pattern="^(low|moderate|high)$")


class WhiteSpace(BaseModel):
    """An identified research gap in the field."""

    description: str
    evidence: str
    cited_as_future_work_in: list[str] = Field(default_factory=list)
    opportunity_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Full result from a literature search query."""

    query: str
    papers: list[Paper] = Field(default_factory=list)
    total_found: int = 0
    summary: str = ""
    field_trend: str = ""
    top_methods: list[str] = Field(default_factory=list)
    white_spaces: list[WhiteSpace] = Field(default_factory=list)
    conflicts: list[ConflictPair] = Field(default_factory=list)

    def top_papers(self, n: int = 5) -> list[Paper]:
        return sorted(self.papers, key=lambda p: p.relevance_score, reverse=True)[:n]

    def negative_results(self) -> list[Paper]:
        return [p for p in self.papers if p.is_negative_result]

    def conflicting_papers(self) -> list[Paper]:
        return [p for p in self.papers if p.conflict_flag]


class ConflictReport(BaseModel):
    """Standalone conflict analysis for a research area."""

    field: str
    conflicts: list[ConflictPair] = Field(default_factory=list)
    summary: str = ""
    most_contested_topic: Optional[str] = None


class WhiteSpaceReport(BaseModel):
    """Standalone white space / gap analysis for a research area."""

    field: str
    white_spaces: list[WhiteSpace] = Field(default_factory=list)
    summary: str = ""
    highest_opportunity: Optional[WhiteSpace] = None


class FieldSummary(BaseModel):
    """High-level intelligence snapshot for a research field."""

    field: str
    paper_count_estimate: int = 0
    growth_trend: str = ""
    dominant_methods: list[str] = Field(default_factory=list)
    key_venues: list[str] = Field(default_factory=list)
    open_problems: list[str] = Field(default_factory=list)
    summary: str = ""
