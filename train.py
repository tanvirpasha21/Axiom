"""
AXIOM Training & Ingestion CLI
================================
Manages two paper stores:

  axiom_papers.jsonl     — Simple JSONL store (always works, no setup)
  axiom_warehouse.jsonl  — Qdrant-backed vector warehouse + Redis cache

QUICK START (no setup required):
    python train.py stats
    python train.py arxiv "fraud detection transformers" 30
    python train.py search "fraud detection"

WITH VECTOR WAREHOUSE (Qdrant + Redis — semantic search):
    # Start services first:
    docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
    docker run -d -p 6379:6379 redis:alpine
    pip install axiom[warehouse]

    python train.py ingest arxiv "fraud detection" 50
    python train.py ingest ieee  "graph neural networks" 50  # needs IEEE_API_KEY
    python train.py ingest all   "transformer fraud"     30  # all sources concurrently
    python train.py wh stats
    python train.py wh search "fraud detection"
    python train.py wh reindex                           # rebuild Qdrant from backup

WITH IEEE (free key from https://developer.ieee.org):
    export IEEE_API_KEY=your-key
    python train.py ingest ieee "fraud detection" 50

COMMANDS:
    ── Simple JSONL store ──────────────────────────────────────
    stats              Show store statistics
    arxiv  QUERY [N]   Fetch N papers from ArXiv into JSONL store
    file   PATH        Import from JSON / JSONL file
    add                Add one paper manually
    list               Print all papers in store
    search QUERY       Keyword search the JSONL store
    remove TITLE       Remove a paper by title
    export PATH        Export store to JSON file
    clear              Wipe the JSONL store

    ── Warehouse (Qdrant + Redis) ──────────────────────────────
    ingest arxiv  QUERY [N]   Async ingest from ArXiv into warehouse
    ingest ieee   QUERY [N]   Async ingest from IEEE Xplore
    ingest all    QUERY [N]   Concurrent ingest from all sources
    wh stats               Warehouse statistics
    wh search QUERY        Semantic search the warehouse
    wh reindex             Rebuild Qdrant index from JSONL backup
    wh cache-clear         Flush Redis search cache
"""

import sys
import os
import json
import asyncio
import tempfile
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
JSONL_PATH = os.getenv("AXIOM_STORE_PATH",     "axiom_papers.jsonl")
WH_PATH    = os.getenv("AXIOM_WAREHOUSE_PATH", "axiom_warehouse.jsonl")
IEEE_KEY   = os.getenv("IEEE_API_KEY",         None)
# ─────────────────────────────────────────────────────────────────────────────


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _store():
    from axiom import LocalPaperStore
    return LocalPaperStore(JSONL_PATH)


def _warehouse():
    from axiom.warehouse import PaperWarehouse
    wh = PaperWarehouse(backup_path=WH_PATH)
    wh.connect()
    return wh


def _print_stats_block(stats: dict):
    print(f"\n  {'─'*50}")
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"      {kk}: {vv}")
        else:
            print(f"  {k:<22}: {v}")
    print(f"  {'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE JSONL STORE COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_stats():
    s = _store()
    st = s.stats()
    print(f"\n  AXIOM JSONL Store — {JSONL_PATH}")
    _print_stats_block(st)
    if st["total"] == 0:
        print("  Store is empty. Start training with:")
        print(f'    python train.py arxiv "your topic" 30')
        print(f'    python train.py file  my_papers.json\n')


def cmd_arxiv(query: str, limit: int = 20):
    from axiom.api_clients import ArxivClient
    print(f'\n  Fetching {limit} papers from ArXiv: "{query}"')
    print("  (ArXiv rate-limits to 1 req/3 sec — this may take a moment)\n")
    client = ArxivClient()
    try:
        papers = client.search(query, limit=limit)
    finally:
        client.close()
    if not papers:
        print("  No papers returned. Try a different query.\n")
        return
    s = _store()
    added = s.add_many(papers)
    skipped = len(papers) - added
    print(f"  Added   : {added} new papers")
    if skipped:
        print(f"  Skipped : {skipped} duplicates")
    print(f"  Total   : {s.stats()['total']} papers in {JSONL_PATH}")
    if added:
        print(f"\n  Sample:")
        for p in papers[:3]:
            print(f"    · [{p.year}] {p.title[:65]}")
    print()


def cmd_file(path: str):
    p = Path(path)
    if not p.exists():
        print(f"\n  Error: file not found: {path}\n")
        return
    s = _store()
    print(f"\n  Importing from: {p.resolve()}")
    added = s.import_from_file(path)
    print(f"  Added  : {added} new papers")
    print(f"  Total  : {s.stats()['total']} papers in {JSONL_PATH}\n")


def cmd_add():
    from axiom.models import Paper
    print("\n  Add a paper manually (Enter to skip optional fields)\n")
    title = input("  Title (required): ").strip()
    if not title:
        print("  Aborted — title is required.\n")
        return
    authors_raw = input("  Authors (comma-separated): ").strip()
    authors = [a.strip() for a in authors_raw.split(",") if a.strip()] if authors_raw else []
    year_raw = input("  Year: ").strip()
    year = int(year_raw) if year_raw.isdigit() else None
    venue    = input("  Venue/Journal: ").strip() or None
    abstract = input("  Abstract: ").strip() or None
    doi      = input("  DOI (e.g. 10.1109/...): ").strip() or None
    url      = input("  URL: ").strip() or None

    paper = Paper(title=title, authors=authors, year=year, venue=venue,
                  abstract=abstract, doi=doi, url=url, source="manual")
    s = _store()
    if s.add(paper):
        print(f"\n  Added: {title}")
        print(f"  Total: {s.stats()['total']} papers\n")
    else:
        print(f"\n  Skipped — already exists.\n")


def cmd_list():
    s = _store()
    papers = s.all_papers()
    if not papers:
        print(f"\n  Store is empty — {JSONL_PATH}\n")
        return
    print(f"\n  {len(papers)} papers in {JSONL_PATH}\n")
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
        rank = f" [{p.quality_rank}]" if p.quality_rank else ""
        print(f"  {i:>4}. {p.title[:65]}{rank}")
        print(f"        {authors} · {p.year or '?'} · {p.venue or p.source}")
    print()


def cmd_search_jsonl(query: str):
    s = _store()
    results = s.search(query, limit=10)
    print(f'\n  Query: "{query}" → {len(results)} results in JSONL store\n')
    for i, p in enumerate(results, 1):
        rank = f" [{p.quality_rank}]" if p.quality_rank else ""
        print(f"  {i}. {p.title}{rank}")
        print(f"     {', '.join(p.authors[:2])} ({p.year}) — {p.venue or p.source}")
        if p.abstract:
            print(f"     {p.abstract[:100]}…")
        print()


def cmd_remove(title: str):
    s = _store()
    if s.remove(title):
        print(f"\n  Removed: {title}")
        print(f"  Total  : {s.stats()['total']} papers\n")
    else:
        print(f'\n  Not found: "{title}"')
        print("  Use  python train.py list  to see exact titles.\n")


def cmd_export(path: str):
    s = _store()
    papers = s.all_papers()
    data = [json.loads(p.model_dump_json()) for p in papers]
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n  Exported {len(papers)} papers to: {path}\n")


def cmd_clear():
    s = _store()
    count = s.stats()["total"]
    if count == 0:
        print("\n  Store is already empty.\n")
        return
    confirm = input(f"\n  Delete all {count} papers from {JSONL_PATH}? Type YES: ").strip()
    if confirm == "YES":
        s.clear()
        print(f"  Cleared — {count} papers removed.\n")
    else:
        print("  Cancelled.\n")


# ══════════════════════════════════════════════════════════════════════════════
# WAREHOUSE COMMANDS (Qdrant + Redis)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_ingest(source: str, query: str, limit: int = 30, year_range=None, quality_only=True):
    """Run async ingestion into the warehouse."""
    async def _run():
        from axiom.ingestion import PaperIngester
        wh = _warehouse()
        ingester = PaperIngester(warehouse=wh, ieee_api_key=IEEE_KEY, quality_only=quality_only)

        print(f'\n  Ingesting [{source.upper()}] "{query}" (limit={limit}, quality_only={quality_only})\n')

        if source == "arxiv":
            result = await ingester.ingest_arxiv(query, limit=limit, year_range=year_range)
            results = [result]
        elif source == "ieee":
            if not IEEE_KEY:
                print("  IEEE_API_KEY not set.")
                print("  Get a free key at https://developer.ieee.org")
                print("  Then: export IEEE_API_KEY=your-key\n")
                wh.close()
                return
            result = await ingester.ingest_ieee(query, limit=limit, year_range=year_range)
            results = [result]
        elif source == "all":
            results = await ingester.ingest_all(query, limit_per_source=limit, year_range=year_range)
        else:
            print(f"  Unknown source: {source}. Use arxiv / ieee / all\n")
            wh.close()
            return

        print()
        for r in results:
            print(f"  [{r.source.upper()}]")
            print(f"    fetched          : {r.fetched}")
            print(f"    after normalise  : {r.after_normalise}")
            print(f"    after dedup      : {r.after_deduplicate}")
            print(f"    after quality    : {r.after_quality_filter}")
            print(f"    stored           : {r.stored}")
            if r.errors:
                for e in r.errors:
                    print(f"    error: {e}")
            print()

        wh.close()

    asyncio.run(_run())


def cmd_wh_stats():
    wh = _warehouse()
    st = wh.stats()
    print(f"\n  AXIOM Warehouse — {WH_PATH}")
    _print_stats_block(st)
    if not st["qdrant_online"]:
        print("  Qdrant is offline. Start with:")
        print("    docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant\n")
    if not st["redis_online"]:
        print("  Redis is offline (optional — caching disabled).")
        print("    docker run -d -p 6379:6379 redis:alpine\n")
    wh.close()


def cmd_wh_search(query: str, quality_only: bool = False):
    wh = _warehouse()
    print(f'\n  Warehouse semantic search: "{query}" (quality_only={quality_only})\n')
    results = wh.search(query, limit=10, quality_only=quality_only)
    if not results:
        print("  No results. Run  python train.py ingest arxiv \"your topic\"  first.\n")
    else:
        for i, p in enumerate(results, 1):
            rank = f" [{p.quality_rank}]" if p.quality_rank else ""
            score = f" {p.relevance_score:.2f}" if p.relevance_score else ""
            print(f"  {i}. [{p.source}{score}]{rank} {p.title}")
            print(f"     {', '.join(p.authors[:2])} ({p.year}) — {p.venue or '?'}")
            if p.abstract:
                print(f"     {p.abstract[:110]}…")
            print()
    wh.close()


def cmd_wh_reindex():
    print("\n  Rebuilding Qdrant index from JSONL backup…")
    wh = _warehouse()
    wh.reindex()
    wh.close()
    print("  Done.\n")


def cmd_wh_cache_clear():
    wh = _warehouse()
    wh.invalidate_cache()
    wh.close()
    print("  Redis cache cleared.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI dispatch
# ══════════════════════════════════════════════════════════════════════════════

def usage():
    print(__doc__)


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        usage()

    # ── Simple JSONL store ────────────────────────────────────────────────────
    elif args[0] == "stats":
        cmd_stats()

    elif args[0] == "arxiv":
        if len(args) < 2:
            print('\n  Usage: python train.py arxiv "query" [limit=20]\n')
        else:
            limit = int(args[2]) if len(args) > 2 and args[2].isdigit() else 20
            cmd_arxiv(args[1], limit=limit)

    elif args[0] == "file":
        if len(args) < 2:
            print('\n  Usage: python train.py file path/to/papers.json\n')
        else:
            cmd_file(args[1])

    elif args[0] == "add":
        cmd_add()

    elif args[0] == "list":
        cmd_list()

    elif args[0] == "search":
        if len(args) < 2:
            print('\n  Usage: python train.py search "query"\n')
        else:
            cmd_search_jsonl(args[1])

    elif args[0] == "remove":
        if len(args) < 2:
            print('\n  Usage: python train.py remove "Exact Paper Title"\n')
        else:
            cmd_remove(" ".join(args[1:]))

    elif args[0] == "export":
        if len(args) < 2:
            print('\n  Usage: python train.py export output.json\n')
        else:
            cmd_export(args[1])

    elif args[0] == "clear":
        cmd_clear()

    # ── Warehouse (async ingest) ──────────────────────────────────────────────
    elif args[0] == "ingest":
        # ingest <source> <query> [limit] [year_range] [--all-quality]
        if len(args) < 3:
            print('\n  Usage: python train.py ingest <arxiv|ieee|all> "query" [limit=30]\n')
        else:
            source = args[1]
            query  = args[2]
            limit  = int(args[3]) if len(args) > 3 and args[3].isdigit() else 30
            year   = args[4] if len(args) > 4 and "-" in args[4] else None
            q_only = "--all-quality" not in args  # default: quality_only=True
            cmd_ingest(source, query, limit=limit, year_range=year, quality_only=q_only)

    # ── Warehouse management ──────────────────────────────────────────────────
    elif args[0] == "wh":
        sub = args[1] if len(args) > 1 else ""
        if sub == "stats":
            cmd_wh_stats()
        elif sub == "search":
            if len(args) < 3:
                print('\n  Usage: python train.py wh search "query" [--all-quality]\n')
            else:
                q_only = "--all-quality" in args
                cmd_wh_search(args[2], quality_only=q_only)
        elif sub == "reindex":
            cmd_wh_reindex()
        elif sub == "cache-clear":
            cmd_wh_cache_clear()
        else:
            print("\n  wh subcommands: stats | search | reindex | cache-clear\n")

    else:
        usage()
