"""
AXIOM Agent Test Suite
======================
Run all offline tests:
    python test_agent.py

Run with live LLM backend (needs Ollama/API running):
    python test_agent.py --live

Run a specific section only:
    python test_agent.py store       # LocalPaperStore tests
    python test_agent.py llm         # LLMPaperClient JSON parsing
    python test_agent.py arxiv       # Live ArXiv (needs internet)
    python test_agent.py agent       # Full agent (needs LLM + internet)

Configure your backend at the top of this file or via env vars:
    AXIOM_BACKEND=ollama AXIOM_MODEL=mistral python test_agent.py --live
"""

import sys
import os
import json
import tempfile
import traceback
from pathlib import Path

# ─── Backend config ───────────────────────────────────────────────────────────
BACKEND      = os.getenv("AXIOM_BACKEND",      "ollama")
MODEL        = os.getenv("AXIOM_MODEL",         "mistral")
API_KEY      = os.getenv("AXIOM_API_KEY",       None)
PAPER_SOURCE = os.getenv("AXIOM_PAPER_SOURCE",  "llm+arxiv")
# ─────────────────────────────────────────────────────────────────────────────

LIVE    = "--live"    in sys.argv
VERBOSE = "--verbose" in sys.argv
SECTION = next((a for a in sys.argv[1:] if not a.startswith("-")), "all")

_passed = _failed = _skipped = 0


# ─── Tiny test runner ─────────────────────────────────────────────────────────

def test(name: str, fn, *, live: bool = False):
    global _passed, _failed, _skipped
    if live and not LIVE:
        print(f"  [SKIP] {name}")
        _skipped += 1
        return
    try:
        fn()
        print(f"  [PASS] {name}")
        _passed += 1
    except Exception as e:
        print(f"  [FAIL] {name} — {e}")
        if VERBOSE:
            traceback.print_exc()
        _failed += 1


def section(title: str, key: str = "all") -> bool:
    if SECTION not in ("all", key):
        return False
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")
    return True


# ─── Assertion helpers ────────────────────────────────────────────────────────

def assert_eq(a, b, msg=""):
    assert a == b, f"{msg} — got {a!r}, expected {b!r}"

def assert_true(v, msg=""):
    assert v, f"{msg} — got {v!r}"

def assert_len(seq, n, msg=""):
    assert len(seq) == n, f"{msg} — len={len(seq)}, expected {n}"

def assert_in(val, container, msg=""):
    assert val in container, f"{msg} — {val!r} not in {container!r}"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def tmp_store():
    """Return a fresh LocalPaperStore backed by a temp file."""
    from axiom import LocalPaperStore
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    f.close()
    Path(f.name).unlink()
    return LocalPaperStore(f.name)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Pydantic models
# ═══════════════════════════════════════════════════════════════════════════════

if section("1. Pydantic models (offline)", "models"):
    from axiom.models import (
        Paper, SearchResult, ConflictPair, WhiteSpace,
        ConflictReport, WhiteSpaceReport, FieldSummary,
    )

    def t_paper_defaults():
        p = Paper(title="T", authors=["A"])
        assert_eq(p.relevance_score, 0.0)
        assert_eq(p.conflict_flag, False)
        assert_eq(p.is_negative_result, False)
        assert_eq(p.source, "semantic_scholar")

    def t_paper_str_two():
        s = str(Paper(title="My Paper", authors=["Alice", "Bob"], year=2023, venue="NeurIPS"))
        assert_in("Alice", s); assert_in("2023", s); assert "et al" not in s

    def t_paper_str_many():
        assert_in("et al.", str(Paper(title="Big", authors=["A","B","C","D"], year=2022)))

    def t_paper_relevance_bounds():
        Paper(title="X", relevance_score=1.0)
        try:
            Paper(title="X", relevance_score=1.5)
            assert False, "Should raise"
        except Exception:
            pass

    def t_search_top_papers():
        papers = [Paper(title="Low", relevance_score=0.2),
                  Paper(title="High", relevance_score=0.9),
                  Paper(title="Mid", relevance_score=0.5)]
        top = SearchResult(query="q", papers=papers).top_papers(2)
        assert_eq(top[0].title, "High")
        assert_eq(top[1].title, "Mid")

    def t_search_filters():
        papers = [Paper(title="A", is_negative_result=True),
                  Paper(title="B", conflict_flag=True),
                  Paper(title="C")]
        sr = SearchResult(query="q", papers=papers)
        assert_len(sr.negative_results(), 1)
        assert_len(sr.conflicting_papers(), 1)

    def t_conflict_severity():
        ConflictPair(paper_a="A", paper_b="B", claim_a="x", claim_b="y",
                     contested_topic="t", severity="high")
        try:
            ConflictPair(paper_a="A", paper_b="B", claim_a="x", claim_b="y",
                         contested_topic="t", severity="bad_value")
            assert False, "Should raise"
        except Exception:
            pass

    def t_whitespace_bounds():
        WhiteSpace(description="gap", evidence="r", opportunity_score=0.8)
        try:
            WhiteSpace(description="gap", evidence="r", opportunity_score=1.5)
            assert False, "Should raise"
        except Exception:
            pass

    test("Paper defaults", t_paper_defaults)
    test("Paper.__str__ two authors", t_paper_str_two)
    test("Paper.__str__ many authors → et al.", t_paper_str_many)
    test("Paper relevance_score out of range raises", t_paper_relevance_bounds)
    test("SearchResult.top_papers ordering", t_search_top_papers)
    test("SearchResult filter methods", t_search_filters)
    test("ConflictPair severity validation", t_conflict_severity)
    test("WhiteSpace opportunity_score bounds", t_whitespace_bounds)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LocalPaperStore
# ═══════════════════════════════════════════════════════════════════════════════

if section("2. LocalPaperStore (offline)", "store"):
    from axiom import LocalPaperStore
    from axiom.models import Paper

    def t_add_and_search():
        s = tmp_store()
        s.add(Paper(title="Fraud Detection with Transformers", authors=["Alice"], year=2023,
                    abstract="We apply transformers to detect fraud in real time.", source="manual"))
        results = s.search("fraud detection")
        assert_len(results, 1)
        assert_eq(results[0].title, "Fraud Detection with Transformers")
        Path(s._path).unlink(missing_ok=True)

    def t_deduplication():
        s = tmp_store()
        p = Paper(title="Dupe", authors=["X"], source="manual")
        assert_true(s.add(p),  "First add → True")
        assert_eq(s.add(p), False, "Second add → False")
        assert_len(s.all_papers(), 1)
        Path(s._path).unlink(missing_ok=True)

    def t_add_many():
        s = tmp_store()
        papers = [Paper(title=f"Paper {i}", source="manual") for i in range(5)]
        assert_eq(s.add_many(papers), 5)
        assert_eq(s.add_many(papers), 0, "Re-adding same papers should add 0")
        Path(s._path).unlink(missing_ok=True)

    def t_persistence():
        path = tempfile.mktemp(suffix=".jsonl")
        s1 = LocalPaperStore(path)
        s1.add(Paper(title="Persistent Paper", authors=["Carol"], source="manual"))
        s2 = LocalPaperStore(path)
        assert_len(s2.all_papers(), 1)
        assert_eq(s2.all_papers()[0].title, "Persistent Paper")
        Path(path).unlink(missing_ok=True)

    def t_year_range():
        s = tmp_store()
        s.add(Paper(title="Old", year=2015, abstract="transformers", source="manual"))
        s.add(Paper(title="New", year=2023, abstract="transformers", source="manual"))
        results = s.search("transformers", year_range="2020-2024")
        assert_len(results, 1)
        assert_eq(results[0].title, "New")
        Path(s._path).unlink(missing_ok=True)

    def t_title_scores_higher():
        s = tmp_store()
        s.add(Paper(title="Fraud Detection Methods",
                    abstract="A generic paper about stuff.", source="manual"))
        s.add(Paper(title="Unrelated Topic",
                    abstract="This paper covers fraud detection methods in depth.", source="manual"))
        results = s.search("fraud detection methods")
        assert_eq(results[0].title, "Fraud Detection Methods", "Title match must rank first")
        Path(s._path).unlink(missing_ok=True)

    def t_import_json_array():
        s = tmp_store()
        data = json.dumps([
            {"title": "Imported A", "authors": ["X"], "year": 2022,
             "abstract": "Something.", "source": "manual"},
            {"title": "Imported B", "authors": ["Y"], "year": 2023,
             "abstract": "Something else.", "source": "manual"},
        ])
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write(data); tmp.close()
        assert_eq(s.import_from_file(tmp.name), 2)
        Path(tmp.name).unlink(missing_ok=True)
        Path(s._path).unlink(missing_ok=True)

    def t_import_jsonl():
        s = tmp_store()
        lines = "\n".join([
            json.dumps({"title": "JSONL 1", "authors": ["A"], "source": "manual"}),
            json.dumps({"title": "JSONL 2", "authors": ["B"], "source": "manual"}),
        ])
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
        tmp.write(lines); tmp.close()
        assert_eq(s.import_from_file(tmp.name), 2)
        Path(tmp.name).unlink(missing_ok=True)
        Path(s._path).unlink(missing_ok=True)

    def t_remove():
        s = tmp_store()
        s.add(Paper(title="Remove Me", source="manual"))
        s.add(Paper(title="Keep Me", source="manual"))
        assert_true(s.remove("Remove Me"))
        assert_len(s.all_papers(), 1)
        assert_eq(s.all_papers()[0].title, "Keep Me")
        Path(s._path).unlink(missing_ok=True)

    def t_clear():
        s = tmp_store()
        s.add_many([Paper(title=f"P{i}", source="manual") for i in range(3)])
        assert_eq(s.clear(), 3)
        assert_len(s.all_papers(), 0)

    def t_stats():
        s = tmp_store()
        s.add(Paper(title="A", year=2021, source="arxiv"))
        s.add(Paper(title="B", year=2023, source="llm_knowledge"))
        st = s.stats()
        assert_eq(st["total"], 2)
        assert_eq(st["year_range"], "2021–2023")
        assert_in("arxiv", st["sources"])
        Path(s._path).unlink(missing_ok=True)

    def t_empty_store_returns_empty():
        s = tmp_store()
        assert_len(s.search("anything"), 0)

    def t_store_via_agent():
        """Agent.store property lets you train without importing LocalPaperStore."""
        from axiom import LiteratureAgent
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        Path(path).unlink(missing_ok=True)

        # We can't construct a full agent without a backend in offline mode,
        # so just verify LocalPaperStore works standalone (agent test is in section 6).
        store = LocalPaperStore(path)
        store.add(Paper(title="Via Agent Store", source="manual"))
        assert_len(store.all_papers(), 1)
        Path(path).unlink(missing_ok=True)

    test("add and keyword search", t_add_and_search)
    test("deduplication", t_deduplication)
    test("add_many + bulk dedup", t_add_many)
    test("persists to disk and reloads", t_persistence)
    test("year_range filter", t_year_range)
    test("title match ranks above body match", t_title_scores_higher)
    test("import_from_file — JSON array", t_import_json_array)
    test("import_from_file — JSONL", t_import_jsonl)
    test("remove paper by title", t_remove)
    test("clear all papers", t_clear)
    test("stats()", t_stats)
    test("empty store returns []", t_empty_store_returns_empty)
    test("LocalPaperStore standalone (offline agent.store proxy)", t_store_via_agent)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LLMPaperClient JSON parsing
# ═══════════════════════════════════════════════════════════════════════════════

if section("3. LLMPaperClient — JSON parsing (offline)", "llm"):
    from axiom.api_clients import LLMPaperClient

    c = LLMPaperClient(backend=None)

    def t_parse_valid():
        raw = json.dumps([
            {"title": "Attention Is All You Need", "authors": ["Vaswani", "Shazeer"],
             "year": 2017, "venue": "NeurIPS",
             "abstract": "Introduces the Transformer.", "citation_count": 50000}
        ])
        papers = c._parse(raw)
        assert_len(papers, 1)
        assert_eq(papers[0].title, "Attention Is All You Need")
        assert_eq(papers[0].source, "llm_knowledge")
        assert_eq(papers[0].year, 2017)

    def t_parse_strips_fences():
        raw = "```json\n" + json.dumps([{"title": "Fenced", "authors": ["X"], "year": 2020}]) + "\n```"
        papers = c._parse(raw)
        assert_len(papers, 1)
        assert_eq(papers[0].title, "Fenced")

    def t_parse_bad_json():
        assert_len(c._parse("not json at all"), 0)

    def t_parse_skips_no_title():
        raw = json.dumps([
            {"authors": ["Nobody"], "year": 2020},
            {"title": "Has Title", "authors": ["Y"], "year": 2021},
        ])
        papers = c._parse(raw)
        assert_len(papers, 1)
        assert_eq(papers[0].title, "Has Title")

    def t_parse_null_fields():
        raw = json.dumps([{"title": "Null Fields", "authors": None, "year": None,
                           "venue": None, "abstract": None, "url": None}])
        papers = c._parse(raw)
        assert_len(papers, 1)
        assert_eq(papers[0].authors, [])

    def t_parse_non_list():
        assert_len(c._parse(json.dumps({"title": "Dict not list"})), 0)

    test("parse valid JSON array", t_parse_valid)
    test("strip markdown fences", t_parse_strips_fences)
    test("invalid JSON → []", t_parse_bad_json)
    test("skip entries without title", t_parse_skips_no_title)
    test("null fields handled gracefully", t_parse_null_fields)
    test("non-list JSON → []", t_parse_non_list)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LiteratureAgent._parse_json
# ═══════════════════════════════════════════════════════════════════════════════

if section("4. LiteratureAgent._parse_json (offline)", "agent"):
    from axiom.literature import LiteratureAgent

    test("plain JSON",           lambda: assert_eq(LiteratureAgent._parse_json('{"x":1}')["x"], 1))
    test("JSON in fences",       lambda: assert_eq(LiteratureAgent._parse_json('```json\n{"x":2}\n```')["x"], 2))
    test("bad JSON returns {}",  lambda: assert_eq(LiteratureAgent._parse_json("bad"), {}))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Live ArXiv (needs internet)
# ═══════════════════════════════════════════════════════════════════════════════

if section("5. ArXiv live search (needs internet)", "arxiv"):
    from axiom.api_clients import ArxivClient

    def t_arxiv_returns_papers():
        client = ArxivClient()
        papers = client.search("fraud detection transformer", limit=3)
        assert_true(len(papers) > 0, "Should return papers")
        assert_true(all(p.title for p in papers), "All have titles")
        assert_true(all(p.source == "arxiv" for p in papers), "source='arxiv'")
        client.close()

    def t_arxiv_has_url():
        client = ArxivClient()
        papers = client.search("deep learning", limit=2)
        assert_true(any(p.url for p in papers), "At least one has URL")
        client.close()

    def t_arxiv_has_year():
        client = ArxivClient()
        papers = client.search("neural network", limit=3)
        assert_true(any(p.year for p in papers), "At least one has year")
        client.close()

    test("returns papers with titles", t_arxiv_returns_papers, live=True)
    test("papers have URLs",           t_arxiv_has_url,          live=True)
    test("papers have year",           t_arxiv_has_year,          live=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Full agent integration (needs LLM backend + internet)
# ═══════════════════════════════════════════════════════════════════════════════

if section("6. Full agent integration (needs LLM + internet)", "agent"):
    from axiom import LiteratureAgent, LocalPaperStore
    from axiom.api_clients import ArxivClient
    from axiom.models import Paper

    def _agent(**kwargs):
        return LiteratureAgent(backend=BACKEND, api_key=API_KEY, model=MODEL, **kwargs)

    def t_search_llm_source():
        agent = _agent(paper_source="llm", verbose=True)
        result = agent.search("attention mechanism transformer", limit=5)
        assert_true(result.summary, "summary not empty")
        assert_true(len(result.papers) > 0, "should return papers")
        print(f"\n         summary: {result.summary[:100]}...")
        print(f"         papers recalled: {len(result.papers)}")
        agent.close()

    def t_search_arxiv_source():
        agent = _agent(paper_source="arxiv", verbose=True)
        result = agent.search("fraud detection neural network", limit=5)
        assert_true(result.summary, "summary not empty")
        agent.close()

    def t_search_llm_plus_arxiv():
        agent = _agent(paper_source="llm+arxiv", verbose=True)
        result = agent.search("graph neural network anomaly", limit=8)
        assert_true(result.summary, "summary not empty")
        sources = {p.source for p in result.papers}
        print(f"\n         sources present: {sources}")
        agent.close()

    def t_search_local_store():
        path = tempfile.mktemp(suffix=".jsonl")
        store = LocalPaperStore(path)
        # Train the store with ArXiv papers
        arxiv = ArxivClient()
        added = store.add_many(arxiv.search("fraud detection machine learning", limit=5))
        assert_true(added > 0, f"ArXiv seeding added {added} papers")
        print(f"\n         trained store with {added} papers")

        agent = _agent(paper_source="local+llm", db_path=path, verbose=True)
        result = agent.search("fraud detection", limit=10)
        assert_true(result.summary, "summary not empty")
        agent.close()
        Path(path).unlink(missing_ok=True)

    def t_find_conflicts():
        agent = _agent(paper_source="llm", verbose=False)
        report = agent.find_conflicts("deep learning vs classical ML for fraud")
        assert_true(report.summary, "conflict report summary not empty")
        print(f"\n         contested topic: {report.most_contested_topic}")
        agent.close()

    def t_find_white_spaces():
        agent = _agent(paper_source="llm", verbose=False)
        report = agent.find_white_spaces("real-time fraud detection streaming")
        assert_true(report.summary, "white space report not empty")
        if report.highest_opportunity:
            print(f"\n         top gap: {report.highest_opportunity.description[:80]}")
        agent.close()

    def t_field_summary():
        agent = _agent(paper_source="llm", verbose=False)
        summary = agent.field_summary("fraud detection in financial transactions")
        assert_true(summary.summary, "field summary not empty")
        assert_true(len(summary.dominant_methods) > 0, "has dominant methods")
        print(f"\n         methods: {', '.join(summary.dominant_methods[:3])}")
        agent.close()

    def t_agent_store_property():
        path = tempfile.mktemp(suffix=".jsonl")
        agent = _agent(paper_source="local+llm", db_path=path, verbose=False)
        added = agent.store.add(Paper(title="My Custom Paper", authors=["Me"],
                                      year=2024, source="manual"))
        assert_true(added, "Should add via agent.store")
        assert_eq(agent.store.stats()["total"], 1)
        agent.close()
        Path(path).unlink(missing_ok=True)

    def t_year_range_filter():
        agent = _agent(paper_source="llm", verbose=False)
        result = agent.search("transformer language model", year_range="2020-2023", limit=5)
        years = [p.year for p in result.papers if p.year]
        out_of_range = [y for y in years if y < 2020 or y > 2023]
        print(f"\n         years returned: {sorted(set(years))}")
        # LLM may not perfectly honour this, so we just warn rather than fail hard
        if out_of_range:
            print(f"         WARNING: {len(out_of_range)} paper(s) outside range: {out_of_range}")
        agent.close()

    test("search — paper_source='llm'",         t_search_llm_source,    live=True)
    test("search — paper_source='arxiv'",       t_search_arxiv_source,  live=True)
    test("search — paper_source='llm+arxiv'",   t_search_llm_plus_arxiv, live=True)
    test("search — paper_source='local+llm'",   t_search_local_store,   live=True)
    test("find_conflicts",                       t_find_conflicts,        live=True)
    test("find_white_spaces",                    t_find_white_spaces,     live=True)
    test("field_summary",                        t_field_summary,         live=True)
    test("agent.store property",                 t_agent_store_property,  live=True)
    test("year_range filter (soft check)",       t_year_range_filter,     live=True)


# ─── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'═' * 58}")
print(f"  {_passed} passed  |  {_failed} failed  |  {_skipped} skipped")
if not LIVE:
    print(f"  Run with --live to also test the LLM backend")
print(f"  Backend: {BACKEND} / {MODEL}  |  source: {PAPER_SOURCE}")
print(f"{'═' * 58}\n")

if _failed:
    sys.exit(1)
