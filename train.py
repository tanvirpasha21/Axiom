"""
AXIOM Training Script
=====================
This file trains your local paper knowledge base (axiom_papers.jsonl).
The more papers you add, the better your agent's answers become.

HOW TO RUN:
    python train.py                    # show current store stats
    python train.py arxiv "your topic" # fetch papers from ArXiv and save
    python train.py file my_papers.json # import from a JSON/JSONL file
    python train.py add                # manually add a single paper
    python train.py clear              # wipe the store and start fresh
    python train.py list               # print all papers in the store

YOUR TRAINING FILE:
    axiom_papers.jsonl  (in this folder)
    Each line is one paper stored as JSON.
    You can open, inspect, and edit it with any text editor.

AFTER TRAINING — use your store in the agent:
    from axiom import LiteratureAgent
    agent = LiteratureAgent(
        backend="ollama",
        model="mistral",
        paper_source="local+llm",       # local store + LLM knowledge
        db_path="axiom_papers.jsonl",   # your training file
    )
    results = agent.search("your topic")
"""

import sys
import json
from pathlib import Path
from axiom import LocalPaperStore
from axiom.api_clients import ArxivClient
from axiom.models import Paper

# ─── Location of your training file ──────────────────────────────────────────
DB_PATH = "axiom_papers.jsonl"
# Change this to use a different file, e.g. "my_domain_papers.jsonl"
# ─────────────────────────────────────────────────────────────────────────────


def load_store() -> LocalPaperStore:
    return LocalPaperStore(DB_PATH)


def cmd_stats(store: LocalPaperStore):
    """Show what's currently in your training file."""
    stats = store.stats()
    print(f"\n  Training file : {stats['db_path']}")
    print(f"  Total papers  : {stats['total']}")
    print(f"  Year range    : {stats['year_range']}")
    print(f"  Sources       : {stats['sources']}")
    if stats["total"] == 0:
        print("\n  Store is empty. Run one of these to start training:")
        print("    python train.py arxiv \"fraud detection transformers\"")
        print("    python train.py file  my_papers.json")
    else:
        print(f"\n  Run  python train.py list  to see all papers.")
    print()


def cmd_arxiv(store: LocalPaperStore, query: str, limit: int = 20):
    """
    Fetch papers from ArXiv and add them to your training file.

    This is the easiest way to seed your store. Run it multiple times
    with different queries to build a rich corpus:

        python train.py arxiv "fraud detection transformers" 30
        python train.py arxiv "graph neural networks anomaly detection" 20
        python train.py arxiv "federated learning privacy finance" 15
    """
    print(f"\n  Fetching up to {limit} papers from ArXiv for: \"{query}\"")
    print("  (This may take a few seconds — ArXiv limits 1 request/3sec)\n")

    client = ArxivClient()
    try:
        papers = client.search(query, limit=limit)
    finally:
        client.close()

    if not papers:
        print("  No papers returned. Try a different query.")
        return

    added = store.add_many(papers)
    skipped = len(papers) - added
    print(f"  Added   : {added} new papers")
    if skipped:
        print(f"  Skipped : {skipped} (already in store)")
    print(f"  Total   : {store.stats()['total']} papers in store")
    print(f"  File    : {DB_PATH}")
    print()

    # Preview what was added
    if added > 0:
        new_papers = [p for p in papers if p.title][:3]
        print("  Sample of added papers:")
        for p in new_papers:
            authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
            print(f"    · {p.title[:65]}")
            print(f"      {authors} ({p.year}) — {p.venue or 'arXiv'}")
        print()


def cmd_file(store: LocalPaperStore, path: str):
    """
    Import papers from a JSON or JSONL file.

    Accepted formats:

    1. JSON array  (e.g. exported from Zotero, Mendeley, or another AXIOM store):
       [
         {"title": "...", "authors": ["..."], "year": 2023, "abstract": "..."},
         {"title": "...", "authors": ["..."], "year": 2022, "abstract": "..."}
       ]

    2. JSONL  (one paper per line — native AXIOM format):
       {"title": "...", "authors": ["..."], "year": 2023, ...}
       {"title": "...", "authors": ["..."], "year": 2022, ...}

    Required fields: title
    Optional fields: authors, year, venue, abstract, url, citation_count

    Example:
        python train.py file my_papers.json
        python train.py file /path/to/exported_library.jsonl
    """
    p = Path(path)
    if not p.exists():
        print(f"\n  Error: file not found: {path}")
        print("  Make sure the path is correct.\n")
        return

    print(f"\n  Importing from: {p.resolve()}")
    added = store.import_from_file(path)
    print(f"  Added  : {added} new papers")
    print(f"  Total  : {store.stats()['total']} papers in store")
    print(f"  File   : {DB_PATH}\n")


def cmd_add(store: LocalPaperStore):
    """
    Manually enter a single paper interactively.

    Useful for adding specific papers you know about that aren't on ArXiv.

    Example:
        python train.py add
    """
    print("\n  Add a paper manually (press Enter to skip optional fields)\n")

    title = input("  Title (required): ").strip()
    if not title:
        print("  Title is required. Aborting.\n")
        return

    authors_raw = input("  Authors (comma-separated, e.g. Smith, Jones): ").strip()
    authors = [a.strip() for a in authors_raw.split(",") if a.strip()] if authors_raw else []

    year_raw = input("  Year (e.g. 2023): ").strip()
    year = int(year_raw) if year_raw.isdigit() else None

    venue = input("  Venue / Journal (e.g. NeurIPS, Nature): ").strip() or None
    abstract = input("  Abstract (paste or type, or leave blank): ").strip() or None
    url = input("  URL (e.g. https://arxiv.org/abs/...): ").strip() or None

    paper = Paper(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        abstract=abstract,
        url=url,
        source="manual",
    )

    added = store.add(paper)
    if added:
        print(f"\n  Added: {title}")
        print(f"  Total: {store.stats()['total']} papers in store\n")
    else:
        print(f"\n  Skipped: a paper with this title already exists.\n")


def cmd_list(store: LocalPaperStore):
    """
    Print all papers currently in your training file.

        python train.py list
    """
    papers = store.all_papers()
    if not papers:
        print("\n  Store is empty.\n")
        return

    print(f"\n  {len(papers)} papers in {DB_PATH}\n")
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
        year = str(p.year) if p.year else "?"
        venue = p.venue or p.source
        print(f"  {i:>4}. {p.title[:65]}")
        print(f"        {authors} · {year} · {venue}")
    print()


def cmd_search(store: LocalPaperStore, query: str):
    """
    Search your local store to verify training quality.

        python train.py search "fraud detection"
    """
    results = store.search(query, limit=10)
    print(f"\n  Query: \"{query}\"")
    print(f"  Found: {len(results)} results in local store\n")
    for i, p in enumerate(results, 1):
        authors = ", ".join(p.authors[:2])
        print(f"  {i}. {p.title}")
        print(f"     {authors} ({p.year}) — {p.venue or p.source}")
        if p.abstract:
            print(f"     {p.abstract[:120]}...")
        print()


def cmd_remove(store: LocalPaperStore, title: str):
    """
    Remove a paper by its exact title.

        python train.py remove "Exact Paper Title Here"
    """
    removed = store.remove(title)
    if removed:
        print(f"\n  Removed: {title}")
        print(f"  Total  : {store.stats()['total']} papers remaining\n")
    else:
        print(f"\n  Not found: \"{title}\"")
        print("  Use  python train.py list  to see exact titles.\n")


def cmd_clear(store: LocalPaperStore):
    """
    Wipe the entire store and start fresh.

        python train.py clear
    """
    count = store.stats()["total"]
    if count == 0:
        print("\n  Store is already empty.\n")
        return
    confirm = input(f"\n  Delete all {count} papers? Type YES to confirm: ").strip()
    if confirm == "YES":
        store.clear()
        print(f"  Cleared. {count} papers removed.\n")
    else:
        print("  Cancelled.\n")


def cmd_export(store: LocalPaperStore, path: str):
    """
    Export your store as a clean JSON array (readable by any tool).

        python train.py export my_backup.json
    """
    papers = store.all_papers()
    data = [json.loads(p.model_dump_json()) for p in papers]
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n  Exported {len(papers)} papers to: {path}\n")


# ─── CLI dispatch ─────────────────────────────────────────────────────────────

COMMANDS = {
    "stats":  "Show training file stats (default)",
    "arxiv":  "python train.py arxiv \"query\" [limit=20]  — fetch from ArXiv",
    "file":   "python train.py file  path/to/file.json     — import from JSON/JSONL",
    "add":    "python train.py add                          — add one paper manually",
    "list":   "python train.py list                         — print all papers",
    "search": "python train.py search \"query\"              — search the store",
    "remove": "python train.py remove \"Paper Title\"        — remove one paper",
    "export": "python train.py export output.json           — export to JSON",
    "clear":  "python train.py clear                        — wipe everything",
}

def usage():
    print("\n  AXIOM Training Script — manage your paper knowledge base\n")
    print(f"  Training file: {DB_PATH}\n")
    print("  Commands:")
    for cmd, desc in COMMANDS.items():
        print(f"    {cmd:<8}  {desc}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    store = load_store()

    if not args or args[0] == "stats":
        cmd_stats(store)

    elif args[0] == "arxiv":
        if len(args) < 2:
            print("\n  Usage: python train.py arxiv \"your query\" [limit]\n")
        else:
            limit = int(args[2]) if len(args) > 2 and args[2].isdigit() else 20
            cmd_arxiv(store, args[1], limit=limit)

    elif args[0] == "file":
        if len(args) < 2:
            print("\n  Usage: python train.py file path/to/file.json\n")
        else:
            cmd_file(store, args[1])

    elif args[0] == "add":
        cmd_add(store)

    elif args[0] == "list":
        cmd_list(store)

    elif args[0] == "search":
        if len(args) < 2:
            print("\n  Usage: python train.py search \"your query\"\n")
        else:
            cmd_search(store, args[1])

    elif args[0] == "remove":
        if len(args) < 2:
            print("\n  Usage: python train.py remove \"Exact Paper Title\"\n")
        else:
            cmd_remove(store, " ".join(args[1:]))

    elif args[0] == "export":
        if len(args) < 2:
            print("\n  Usage: python train.py export output.json\n")
        else:
            cmd_export(store, args[1])

    elif args[0] == "clear":
        cmd_clear(store)

    else:
        usage()
