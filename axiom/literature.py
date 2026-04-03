"""
AXIOM Literature Agent
======================
The primary agent. Orchestrates paper retrieval, Claude-powered synthesis,
conflict detection, and white space identification.

Usage:
    from axiom import LiteratureAgent

    agent = LiteratureAgent()                         # uses ANTHROPIC_API_KEY from env
    agent = LiteratureAgent(api_key="sk-ant-...")     # or pass directly

    # Search and synthesise
    result = agent.search("fraud detection transformers fintech")
    print(result.summary)
    for paper in result.top_papers(5):
        print(paper)

    # Detect conflicts in a field
    report = agent.find_conflicts("graph neural networks fraud detection")
    print(report.summary)

    # Surface white spaces
    gaps = agent.find_white_spaces("real-time fraud detection streaming")
    print(gaps.highest_opportunity.description)

    # Field-level intelligence
    snapshot = agent.field_summary("fintech fraud detection")
    print(snapshot.summary)
"""

from __future__ import annotations

import os
import json
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from .models import (
    Paper,
    SearchResult,
    ConflictReport,
    WhiteSpaceReport,
    FieldSummary,
    ConflictPair,
    WhiteSpace,
)
from .api_clients import SemanticScholarClient, ArxivClient
from .llm_backends import get_backend, LLMBackend

console = Console()

SYNTHESIS_SYSTEM = """You are AXIOM, an expert research intelligence agent.
You receive a list of academic papers (title, abstract, year, venue, citations) and a research query.
Your job is to:
1. Write a sharp 2-3 sentence synthesis of what the field currently says about this query
2. Identify the dominant methods and approaches
3. Detect any conflicting claims between papers — papers that directly contradict each other
4. Surface white spaces: questions that are cited as future work but unaddressed
5. Flag negative results (papers reporting what does NOT work)

Always respond in valid JSON matching the schema provided. Be specific, not generic.
Never hallucinate paper titles or authors — only use what you were given."""

CONFLICT_SYSTEM = """You are AXIOM conflict radar.
Given a list of papers, identify pairs that make opposing or contradictory empirical claims.
Focus on methodological disagreements and contradictory results, not just different approaches.
Respond in valid JSON only."""

WHITE_SPACE_SYSTEM = """You are AXIOM gap finder.
Given papers in a research area, identify specific research questions that:
- Multiple papers cite as future work, OR
- Are conspicuously absent given the field's progression, OR
- Have been attempted and failed, leaving the problem open

Each white space should be a concrete, actionable research direction.
Respond in valid JSON only."""

FIELD_SUMMARY_SYSTEM = """You are AXIOM field intelligence.
Provide a structured overview of a research field based on the papers provided.
Include growth trends, dominant methods, key venues, and open problems.
Be analytical and specific. Respond in valid JSON only."""


class LiteratureAgent:
    """
    AXIOM Literature Agent.

    Searches Semantic Scholar + ArXiv, then uses an LLM to synthesise findings,
    detect conflicts, and surface white spaces — all locally on your machine.

    Args:
        backend: LLM backend to use. "anthropic" (default), "ollama", or "openai".
        api_key: API key for the backend (ANTHROPIC_API_KEY or OPENAI_API_KEY env var).
        model: Model name. E.g. "claude-opus-4-5", "llama2", "mistral", "gpt-4".
        ss_api_key: Semantic Scholar API key (optional, for higher rate limits).
        base_url: Base URL for OpenAI-compatible APIs (e.g., local llama.cpp server).
        verbose: Show rich progress output. Default True.
        max_papers: Max papers to fetch per search. Default 20.
    """

    def __init__(
        self,
        backend: str = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        ss_api_key: str | None = None,
        base_url: str | None = None,
        verbose: bool = True,
        max_papers: int = 20,
    ):
        self._llm = get_backend(
            backend_type=backend,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        self._ss = SemanticScholarClient(api_key=ss_api_key)
        self._arxiv = ArxivClient()
        self.verbose = verbose
        self.max_papers = max_papers

    def search(
        self,
        query: str,
        year_range: str | None = None,
        include_arxiv: bool = True,
        limit: int | None = None,
    ) -> SearchResult:
        """
        Search for papers and return a fully synthesised SearchResult.

        Args:
            query: Natural language research question or keywords.
            year_range: Optional year filter e.g. "2020-2024" or "2023-".
            include_arxiv: Also search ArXiv preprints. Default True.
            limit: Override max_papers for this call.

        Returns:
            SearchResult with papers, summary, conflicts, and white spaces.
        """
        n = limit or self.max_papers

        with self._progress(f"Searching literature for: [bold]{query}[/bold]") as progress:
            task = progress.add_task("Querying Semantic Scholar...", total=None)

            papers = self._ss.search(query, limit=n, year_range=year_range)

            if include_arxiv:
                progress.update(task, description="Querying ArXiv preprints...")
                arxiv_papers = self._arxiv.search(query, limit=min(n // 2, 10))
                existing_titles = {p.title.lower() for p in papers}
                for ap in arxiv_papers:
                    if ap.title.lower() not in existing_titles:
                        papers.append(ap)

            progress.update(task, description=f"Synthesising {len(papers)} papers with Claude...")
            result = self._synthesise(query, papers)

        if self.verbose:
            self._print_result(result)

        return result

    def find_conflicts(self, query: str) -> ConflictReport:
        """
        Search a field and return a detailed conflict analysis.

        Args:
            query: Research area to analyse for conflicting claims.

        Returns:
            ConflictReport listing opposing papers and contested topics.
        """
        with self._progress(f"Running conflict radar on: [bold]{query}[/bold]") as progress:
            progress.add_task("Fetching papers...", total=None)
            papers = self._ss.search(query, limit=self.max_papers)

        report = self._detect_conflicts(query, papers)

        if self.verbose:
            self._print_conflicts(report)

        return report

    def find_white_spaces(self, query: str) -> WhiteSpaceReport:
        """
        Search a field and return identified research gaps.

        Args:
            query: Research area to scan for white spaces.

        Returns:
            WhiteSpaceReport with prioritised research opportunities.
        """
        with self._progress(f"Scanning for white spaces in: [bold]{query}[/bold]") as progress:
            progress.add_task("Fetching papers...", total=None)
            papers = self._ss.search(query, limit=self.max_papers)

        report = self._find_gaps(query, papers)

        if self.verbose:
            self._print_gaps(report)

        return report

    def field_summary(self, field: str) -> FieldSummary:
        """
        Return a high-level intelligence snapshot of a research field.

        Args:
            field: Research field or topic.

        Returns:
            FieldSummary with trends, methods, venues, and open problems.
        """
        with self._progress(f"Analysing field: [bold]{field}[/bold]") as progress:
            progress.add_task("Fetching representative papers...", total=None)
            papers = self._ss.search(field, limit=self.max_papers)

        summary = self._summarise_field(field, papers)

        if self.verbose:
            self._print_field_summary(summary)

        return summary

    def _synthesise(self, query: str, papers: list[Paper]) -> SearchResult:
        """Call Claude to synthesise papers into a structured SearchResult."""
        papers_payload = self._papers_to_payload(papers)

        prompt = f"""Research query: {query}

Papers retrieved ({len(papers)} total):
{papers_payload}

Return a JSON object with this exact schema:
{{
  "summary": "2-3 sentence synthesis of what the field says about this query",
  "field_trend": "one sentence on how this area is evolving",
  "top_methods": ["method1", "method2", "method3"],
  "paper_signals": [
    {{
      "title": "exact paper title from the list",
      "relevance_score": 0.0-1.0,
      "tags": ["tag1", "tag2"],
      "conflict_flag": true/false,
      "supports_hypothesis": true/false,
      "is_negative_result": true/false
    }}
  ],
  "conflicts": [
    {{
      "paper_a": "title of first paper",
      "paper_b": "title of second paper",
      "claim_a": "what paper A claims",
      "claim_b": "what paper B claims",
      "contested_topic": "the topic in dispute",
      "severity": "low|moderate|high"
    }}
  ],
  "white_spaces": [
    {{
      "description": "specific unaddressed research question",
      "evidence": "why this is a gap",
      "cited_as_future_work_in": ["paper title"],
      "opportunity_score": 0.0-1.0
    }}
  ]
}}"""

        raw = self._ask_claude(SYNTHESIS_SYSTEM, prompt)
        data = self._parse_json(raw)

        signals = {s["title"].lower(): s for s in data.get("paper_signals", [])}
        for paper in papers:
            sig = signals.get(paper.title.lower(), {})
            paper.relevance_score = sig.get("relevance_score", 0.5)
            paper.tags = sig.get("tags", [])
            paper.conflict_flag = sig.get("conflict_flag", False)
            paper.supports_hypothesis = sig.get("supports_hypothesis", False)
            paper.is_negative_result = sig.get("is_negative_result", False)

        conflicts = [ConflictPair(**c) for c in data.get("conflicts", [])]
        white_spaces = [WhiteSpace(**w) for w in data.get("white_spaces", [])]

        return SearchResult(
            query=query,
            papers=papers,
            total_found=len(papers),
            summary=data.get("summary", ""),
            field_trend=data.get("field_trend", ""),
            top_methods=data.get("top_methods", []),
            white_spaces=white_spaces,
            conflicts=conflicts,
        )

    def _detect_conflicts(self, field: str, papers: list[Paper]) -> ConflictReport:
        papers_payload = self._papers_to_payload(papers)

        prompt = f"""Research area: {field}

Papers:
{papers_payload}

Return JSON:
{{
  "summary": "overall picture of disagreement in this field",
  "most_contested_topic": "the single most disputed topic",
  "conflicts": [
    {{
      "paper_a": "title",
      "paper_b": "title",
      "claim_a": "what it claims",
      "claim_b": "what the other claims",
      "contested_topic": "topic",
      "severity": "low|moderate|high"
    }}
  ]
}}"""

        raw = self._ask_claude(CONFLICT_SYSTEM, prompt)
        data = self._parse_json(raw)

        return ConflictReport(
            field=field,
            summary=data.get("summary", ""),
            most_contested_topic=data.get("most_contested_topic"),
            conflicts=[ConflictPair(**c) for c in data.get("conflicts", [])],
        )

    def _find_gaps(self, field: str, papers: list[Paper]) -> WhiteSpaceReport:
        papers_payload = self._papers_to_payload(papers)

        prompt = f"""Research area: {field}

Papers:
{papers_payload}

Return JSON:
{{
  "summary": "overview of where the field has gaps",
  "white_spaces": [
    {{
      "description": "specific research gap or open question",
      "evidence": "concrete evidence this is unaddressed",
      "cited_as_future_work_in": ["paper title"],
      "opportunity_score": 0.0-1.0
    }}
  ]
}}"""

        raw = self._ask_claude(WHITE_SPACE_SYSTEM, prompt)
        data = self._parse_json(raw)

        white_spaces = [WhiteSpace(**w) for w in data.get("white_spaces", [])]
        highest = max(white_spaces, key=lambda w: w.opportunity_score) if white_spaces else None

        return WhiteSpaceReport(
            field=field,
            summary=data.get("summary", ""),
            white_spaces=white_spaces,
            highest_opportunity=highest,
        )

    def _summarise_field(self, field: str, papers: list[Paper]) -> FieldSummary:
        papers_payload = self._papers_to_payload(papers)

        prompt = f"""Field: {field}

Papers:
{papers_payload}

Return JSON:
{{
  "paper_count_estimate": integer estimate of total papers in field,
  "growth_trend": "sentence on citation/publication growth",
  "dominant_methods": ["method1", "method2", "method3"],
  "key_venues": ["venue1", "venue2"],
  "open_problems": ["problem1", "problem2", "problem3"],
  "summary": "3-4 sentence expert overview of the field's current state"
}}"""

        raw = self._ask_claude(FIELD_SUMMARY_SYSTEM, prompt)
        data = self._parse_json(raw)

        return FieldSummary(
            field=field,
            paper_count_estimate=data.get("paper_count_estimate", 0),
            growth_trend=data.get("growth_trend", ""),
            dominant_methods=data.get("dominant_methods", []),
            key_venues=data.get("key_venues", []),
            open_problems=data.get("open_problems", []),
            summary=data.get("summary", ""),
        )

    def _ask_claude(self, system: str, prompt: str) -> str:
        return self._llm.query(system, prompt, max_tokens=2048)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _papers_to_payload(papers: list[Paper]) -> str:
        lines = []
        for i, p in enumerate(papers, 1):
            authors = ", ".join(p.authors[:3])
            abstract_snippet = (p.abstract or "")[:200].replace("\n", " ")
            lines.append(
                f"{i}. [{p.title}] — {authors} ({p.year}, {p.venue})\n"
                f"   Citations: {p.citation_count or 'unknown'}\n"
                f"   Abstract: {abstract_snippet}..."
            )
        return "\n\n".join(lines)

    def _progress(self, description: str):
        if self.verbose:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console,
            )
        return _NullProgress()

    def _print_result(self, result: SearchResult):
        console.print()
        console.print(Panel(
            Text(result.summary, style="white"),
            title=f"[bold #7F77DD]AXIOM · Search results for '{result.query}'[/bold #7F77DD]",
            border_style="#534AB7",
        ))

        if result.field_trend:
            console.print(f"  [dim]Trend:[/dim] {result.field_trend}")

        console.print(f"\n  [bold]Top papers[/bold] ({len(result.papers)} retrieved)\n")
        for paper in result.top_papers(5):
            score_bar = "█" * int(paper.relevance_score * 10)
            flags = ""
            if paper.conflict_flag:
                flags += " [red][conflict][/red]"
            if paper.is_negative_result:
                flags += " [orange3][negative][/orange3]"
            if paper.supports_hypothesis:
                flags += " [green][supports][/green]"
            console.print(
                f"  [bold]{paper.title}[/bold]{flags}\n"
                f"  [dim]{', '.join(paper.authors[:2])} · {paper.year} · {paper.venue}[/dim]\n"
                f"  [#7F77DD]{score_bar}[/#7F77DD] {paper.relevance_score:.0%} relevance\n"
            )

        if result.white_spaces:
            console.print(f"  [bold #BA7517]White spaces identified: {len(result.white_spaces)}[/bold #BA7517]")
            for ws in result.white_spaces[:2]:
                console.print(f"  [#BA7517]◉[/#BA7517] {ws.description}")

        if result.conflicts:
            console.print(f"\n  [bold #D85A30]Conflicts detected: {len(result.conflicts)}[/bold #D85A30]")
            for c in result.conflicts[:2]:
                console.print(f"  [#D85A30]✕[/#D85A30] {c.contested_topic} ({c.severity} severity)")

        console.print()

    def _print_conflicts(self, report: ConflictReport):
        console.print()
        console.print(Panel(
            Text(report.summary),
            title=f"[bold #D85A30]AXIOM · Conflict Radar · {report.field}[/bold #D85A30]",
            border_style="#993C1D",
        ))
        for c in report.conflicts:
            console.print(f"\n  [bold #D85A30]{c.contested_topic}[/bold #D85A30] ({c.severity})")
            console.print(f"  [dim]A:[/dim] {c.paper_a}")
            console.print(f"    → {c.claim_a}")
            console.print(f"  [dim]B:[/dim] {c.paper_b}")
            console.print(f"    → {c.claim_b}")
        console.print()

    def _print_gaps(self, report: WhiteSpaceReport):
        console.print()
        console.print(Panel(
            Text(report.summary),
            title=f"[bold #BA7517]AXIOM · White Space Finder · {report.field}[/bold #BA7517]",
            border_style="#854F0B",
        ))
        for ws in sorted(report.white_spaces, key=lambda w: w.opportunity_score, reverse=True):
            bar = "█" * int(ws.opportunity_score * 10)
            console.print(f"\n  [#BA7517]{bar}[/#BA7517] {ws.opportunity_score:.0%}")
            console.print(f"  [bold]{ws.description}[/bold]")
            console.print(f"  [dim]{ws.evidence}[/dim]")
        console.print()

    def _print_field_summary(self, summary: FieldSummary):
        console.print()
        console.print(Panel(
            Text(summary.summary),
            title=f"[bold #1D9E75]AXIOM · Field Intelligence · {summary.field}[/bold #1D9E75]",
            border_style="#0F6E56",
        ))
        console.print(f"  [dim]~{summary.paper_count_estimate:,} papers[/dim]  {summary.growth_trend}")
        console.print(f"\n  [bold]Dominant methods:[/bold] {', '.join(summary.dominant_methods)}")
        console.print(f"  [bold]Key venues:[/bold] {', '.join(summary.key_venues)}")
        console.print(f"\n  [bold]Open problems:[/bold]")
        for p in summary.open_problems:
            console.print(f"  [#1D9E75]·[/#1D9E75] {p}")
        console.print()

    def close(self):
        self._llm.close()
        self._ss.close()
        self._arxiv.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _NullProgress:
    """Silent no-op progress context when verbose=False."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def add_task(self, *_, **__): return 0
    def update(self, *_, **__): pass
