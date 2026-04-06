"""
AXIOM Async Ingestion Pipeline
================================
Orchestrates concurrent paper fetching from multiple sources,
normalisation, deduplication, quality filtering, and warehouse storage.

Pipeline per source:
    fetch (async, concurrent) → normalise → deduplicate → quality filter → store

Usage:
    import asyncio
    from axiom.ingestion import PaperIngester
    from axiom.warehouse import PaperWarehouse

    wh = PaperWarehouse().connect()
    ingester = PaperIngester(warehouse=wh)

    # Single source
    result = asyncio.run(ingester.ingest_arxiv("fraud detection transformers", limit=50))

    # All sources concurrently
    results = asyncio.run(ingester.ingest_all(
        query="fraud detection transformers",
        limit_per_source=30,
        quality_only=True,
    ))

    for r in results:
        print(r)

    wh.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from .models import Paper, IngestResult
from .normaliser import Normaliser
from .quality import is_high_quality, enrich_quality
from .warehouse import PaperWarehouse

logger = logging.getLogger(__name__)


class PaperIngester:
    """
    Async ingestion pipeline.

    All public ingest_* methods are coroutines — run with asyncio.run()
    or await inside an async context.

    Args:
        warehouse:    Connected PaperWarehouse instance.
        ieee_api_key: IEEE Xplore API key (optional — skipped if not set).
        quality_only: If True, only store Q1/Q2/top_conference papers.
    """

    def __init__(
        self,
        warehouse: PaperWarehouse,
        ieee_api_key: Optional[str] = None,
        quality_only: bool = True,
    ):
        self._wh           = warehouse
        self._ieee_key     = ieee_api_key or os.getenv("IEEE_API_KEY")
        self._quality_only = quality_only
        self._norm         = Normaliser()

    # ──────────────────────────────────────────────────────────────────────────
    # Public ingestion methods
    # ──────────────────────────────────────────────────────────────────────────

    async def ingest_arxiv(
        self,
        query: str,
        limit: int = 50,
        year_range: Optional[str] = None,
        quality_only: Optional[bool] = None,
    ) -> IngestResult:
        """
        Fetch papers from ArXiv and store them.

        Note: ArXiv papers are preprints — many won't have a Q1/Q2 journal
        venue. When quality_only=True, ArXiv papers are stored only if
        their venue classifies as high quality (common for cross-posted papers).
        Use quality_only=False if you want all arXiv preprints.
        """
        from .sources.arxiv import AsyncArxivClient
        result = IngestResult(source="arxiv", query=query)
        q_only = quality_only if quality_only is not None else self._quality_only

        async with AsyncArxivClient() as client:
            raw = await client.search(query, limit=limit, year_range=year_range)

        result.fetched = len(raw)
        return await self._process(raw, result, q_only)

    async def ingest_ieee(
        self,
        query: str,
        limit: int = 50,
        year_range: Optional[str] = None,
        quality_only: Optional[bool] = None,
        content_types: Optional[list[str]] = None,
    ) -> IngestResult:
        """
        Fetch papers from IEEE Xplore and store them.

        IEEE papers are predominantly Q1/Q2 journal and top conference
        publications — quality filter has high hit rate here.

        Requires IEEE_API_KEY env var or ieee_api_key= constructor param.
        Free key: https://developer.ieee.org
        """
        from .sources.ieee import IEEEClient
        result = IngestResult(source="ieee", query=query)
        q_only = quality_only if quality_only is not None else self._quality_only

        if not self._ieee_key:
            msg = (
                "IEEE_API_KEY not set. "
                "Get a free key at https://developer.ieee.org "
                "then set: export IEEE_API_KEY=your-key"
            )
            logger.warning(msg)
            result.errors.append(msg)
            return result

        try:
            async with IEEEClient(api_key=self._ieee_key) as client:
                raw = await client.search(
                    query,
                    limit=limit,
                    year_range=year_range,
                    content_types=content_types,
                )
        except ValueError as e:
            result.errors.append(str(e))
            return result

        result.fetched = len(raw)
        return await self._process(raw, result, q_only)

    async def ingest_all(
        self,
        query: str,
        limit_per_source: int = 30,
        year_range: Optional[str] = None,
        quality_only: Optional[bool] = None,
        sources: Optional[list[str]] = None,
    ) -> list[IngestResult]:
        """
        Concurrently ingest from all available sources.

        Args:
            query:            Search query.
            limit_per_source: Max papers per source.
            year_range:       Optional "YYYY-YYYY" filter.
            quality_only:     Override instance-level quality_only setting.
            sources:          List of sources to use: ["arxiv", "ieee"].
                              Defaults to all available (IEEE only if key is set).

        Returns:
            List of IngestResult — one per source attempted.
        """
        if sources is None:
            sources = ["arxiv"]
            if self._ieee_key:
                sources.append("ieee")

        tasks = []
        labels = []

        if "arxiv" in sources:
            tasks.append(self.ingest_arxiv(
                query, limit=limit_per_source,
                year_range=year_range, quality_only=quality_only,
            ))
            labels.append("arxiv")

        if "ieee" in sources and self._ieee_key:
            tasks.append(self.ingest_ieee(
                query, limit=limit_per_source,
                year_range=year_range, quality_only=quality_only,
            ))
            labels.append("ieee")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        ingest_results: list[IngestResult] = []
        for label, res in zip(labels, results):
            if isinstance(res, Exception):
                logger.error(f"Ingestion error [{label}]: {res}")
                ingest_results.append(IngestResult(source=label, query=query,
                                                   errors=[str(res)]))
            else:
                ingest_results.append(res)

        total_stored = sum(r.stored for r in ingest_results)
        logger.info(
            f"ingest_all('{query}'): "
            f"stored {total_stored} new papers across {len(ingest_results)} source(s)"
        )
        return ingest_results

    async def ingest_papers(
        self,
        papers: list[Paper],
        source_label: str = "manual",
        quality_only: Optional[bool] = None,
    ) -> IngestResult:
        """
        Ingest a list of Paper objects directly (e.g. from a JSON export).

        Useful for importing from Zotero, Mendeley, or any other tool.
        """
        result = IngestResult(source=source_label, query="<direct import>")
        result.fetched = len(papers)
        q_only = quality_only if quality_only is not None else self._quality_only
        return await self._process(papers, result, q_only)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal pipeline
    # ──────────────────────────────────────────────────────────────────────────

    async def _process(
        self,
        raw: list[Paper],
        result: IngestResult,
        quality_only: bool,
    ) -> IngestResult:
        """
        Shared pipeline: normalise → deduplicate → quality filter → store.
        Runs synchronously inside the coroutine (CPU-bound, fast enough).
        """
        if not raw:
            return result

        # 1. Normalise (cleans fields, enriches quality_rank)
        normalised = self._norm.normalise_many(raw)
        result.after_normalise = len(normalised)

        # 2. Deduplicate against warehouse
        unique, _ = self._norm.deduplicate(
            normalised,
            existing_fingerprints=self._wh._fingerprints,
        )
        result.after_deduplicate = len(unique)

        # 3. Quality filter
        if quality_only:
            high_q = [p for p in unique if is_high_quality(p)]
            unranked = len(unique) - len(high_q)
            if unranked:
                logger.debug(
                    f"[{result.source}] quality filter: dropped {unranked} unranked papers"
                )
            result.after_quality_filter = len(high_q)
        else:
            high_q = unique
            result.after_quality_filter = len(unique)

        if not high_q:
            logger.info(f"[{result.source}] nothing new to store after filtering")
            return result

        # 4. Store — warehouse handles its own dedup check too (safety net)
        stored = self._wh.store(high_q)
        result.stored = stored

        logger.info(str(result))
        return result
