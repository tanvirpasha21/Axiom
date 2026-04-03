# AXIOM — Agentic Research Intelligence Platform

**Search, synthesise, detect conflicts, surface white spaces — powered by Claude + Semantic Scholar + ArXiv.**

Built by [MD Tanvir Anjum](mailto:contact@voidstudio.tech) · [Void Studio](https://voidstudio.tech)

---

## What It Does

AXIOM is a locally-runnable Python agent that goes far beyond listing papers. It retrieves academic literature from Semantic Scholar and ArXiv, then uses a Claude-class LLM to reason across papers and produce structured intelligence.

| Capability | What you get |
|---|---|
| **Search & Synthesise** | What the field *collectively says*, not just a ranked list |
| **Conflict Detection** | Pairs of papers with directly contradictory empirical claims |
| **White Space Detection** | Research questions cited as future work but never addressed |
| **Negative Result Flagging** | Papers reporting what *doesn't* work — the most underused signal |
| **Field Intelligence** | Growth trends, dominant methods, key venues, open problems |

---

## Quick Start

```python
from axiom import LiteratureAgent

# Free local option — no API key needed
agent = LiteratureAgent(backend="ollama", model="mistral")

results = agent.search("fraud detection transformer models")
print(results.summary)
print(results.top_papers(5))
```

---

## Installation

### From Source

```bash
git clone https://github.com/tanvirpasha21/axiom-literature-agent
cd axiom-literature-agent
pip install -e .
```

### With Optional Backends

```bash
pip install -e ".[ollama]"    # Local Ollama support
pip install -e ".[openai]"    # OpenAI / OpenRouter support
pip install -e ".[all]"       # All backends
pip install -e ".[dev]"       # + pytest, ruff, mypy
```

---

## Backends

### Ollama — Free, Local, No API Key

Best for development and privacy-sensitive workloads. Runs entirely on your machine.

```bash
# 1. Install Ollama — https://ollama.com
# 2. Pull a model (one-time download)
ollama pull mistral      # Recommended: fast + high quality (7B)
ollama pull llama2       # Classic option (7B)
ollama pull neural-chat  # Optimised for instruction-following (7B)
# 3. Start the server
ollama serve
```

```python
agent = LiteratureAgent(backend="ollama", model="mistral")
```

### Anthropic (Claude)

```python
agent = LiteratureAgent(backend="anthropic", api_key="sk-ant-...")
# or: export ANTHROPIC_API_KEY=sk-ant-...
```

Available models: `claude-opus-4-5` (default), `claude-sonnet-4-20250514`, `claude-haiku-3-5`

### OpenRouter — 200+ Models via One Key

```python
agent = LiteratureAgent(
    backend="openrouter",
    api_key="sk-or-...",
    model="openai/gpt-4o"
)
# or: export OPENROUTER_API_KEY=sk-or-...
```

### OpenAI

```python
agent = LiteratureAgent(backend="openai", api_key="sk-...", model="gpt-4o")
# or: export OPENAI_API_KEY=sk-...
```

Available models: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-3.5-turbo`

### Local OpenAI-Compatible Server (llama.cpp, vLLM)

```python
agent = LiteratureAgent(
    backend="openai",
    model="local-model",
    api_key="not-needed",
    base_url="http://localhost:8000"
)
```

---

## Environment Variables

Create a `.env` file (add to `.gitignore`):

```bash
# Set the backend and model globally
AXIOM_BACKEND=ollama
AXIOM_MODEL=mistral
AXIOM_BASE_URL=http://localhost:8000   # for local OpenAI-compatible servers

# LLM API keys (only set the one you use)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...

# Semantic Scholar key — see rate limit section below
SEMANTIC_SCHOLAR_API_KEY=your-key-here
```

---

## Python API

### Initialisation

```python
LiteratureAgent(
    backend="anthropic",  # "anthropic" | "ollama" | "openai" | "openrouter"
    api_key=None,         # LLM key (or use env var)
    model=None,           # Model name (sensible defaults per backend)
    ss_api_key=None,      # Semantic Scholar API key (see rate limit section)
    base_url=None,        # For OpenAI-compatible local servers
    verbose=True,         # Rich terminal output with panels and progress
    max_papers=20,        # Max papers to retrieve per query
)
```

### `search` — Retrieve and Synthesise

```python
results = agent.search(
    "graph neural networks for fraud detection",
    year_range="2021-2024",   # Optional date filter
    include_arxiv=True,        # Also query ArXiv preprints
    limit=20
)

results.summary                # 2-3 sentence synthesis of what the field says
results.field_trend            # One-sentence evolution of the area
results.top_methods            # List of dominant approaches
results.top_papers(n=5)        # Papers sorted by Claude-assigned relevance score
results.negative_results()     # Papers flagged as reporting what doesn't work
results.conflicting_papers()   # Papers flagged as part of a conflict
results.white_spaces           # List[WhiteSpace] — research gaps identified
results.conflicts              # List[ConflictPair] — opposing claims
```

### `find_conflicts` — Detect Contradictions

```python
report = agent.find_conflicts("attention mechanisms in NLP")

report.summary
report.most_contested_topic
for c in report.conflicts:
    print(f"{c.paper_a} vs {c.paper_b}")
    print(f"  Claim A: {c.claim_a}")
    print(f"  Claim B: {c.claim_b}")
    print(f"  Severity: {c.severity}")   # "low" | "moderate" | "high"
```

### `find_white_spaces` — Surface Research Gaps

```python
report = agent.find_white_spaces("real-time fraud detection streaming")

report.summary
report.highest_opportunity     # WhiteSpace with highest opportunity_score
for gap in report.white_spaces:
    print(f"{gap.description}  [{gap.opportunity_score:.0%}]")
    print(f"  Cited as future work in: {gap.cited_as_future_work_in}")
```

### `field_summary` — Field Intelligence Snapshot

```python
snapshot = agent.field_summary("fintech fraud detection")

snapshot.growth_trend
snapshot.dominant_methods    # List[str]
snapshot.key_venues          # List[str] — top conferences and journals
snapshot.open_problems       # List[str]
snapshot.summary             # 3-4 sentence expert overview
```

---

## CLI

The `axiom` command is available after installation:

```bash
# Search and synthesise
axiom search "fraud detection transformers" \
  --year 2020-2024 \
  --backend ollama \
  --model mistral \
  --limit 20

# Detect conflicting claims
axiom conflicts "graph neural networks" --backend ollama

# Find research gaps
axiom gaps "real-time fraud detection" --backend anthropic

# Field intelligence
axiom field "fintech fraud detection" --backend openrouter --model openai/gpt-4o

# Set default backend/model via env and run without flags
export AXIOM_BACKEND=ollama
export AXIOM_MODEL=mistral
axiom search "drug discovery language models"
```

**Flags available on all commands:**

| Flag | Description |
|---|---|
| `--backend` | LLM backend (`anthropic`, `ollama`, `openai`, `openrouter`) |
| `--model` | Model name |
| `--api-key` | API key (overrides env var) |
| `--year` | Year range, e.g. `2020-2024` |
| `--limit` | Number of papers to retrieve (default: 20) |
| `--no-arxiv` | Skip ArXiv, use Semantic Scholar only |

---

## Data Models

### `Paper`

```python
paper.title
paper.authors            # List[str]
paper.year
paper.venue
paper.abstract
paper.url
paper.citation_count
paper.relevance_score    # float 0.0–1.0, assigned by Claude
paper.tags               # List[str], assigned by Claude
paper.conflict_flag      # bool — part of a detected conflict
paper.is_negative_result # bool — reports what doesn't work
paper.source             # "semantic_scholar" or "arxiv"
```

### `WhiteSpace`

```python
ws.description                # Specific research question
ws.evidence                   # Why this is a gap
ws.cited_as_future_work_in    # List of paper titles mentioning it
ws.opportunity_score          # float 0.0–1.0
```

### `ConflictPair`

```python
cp.paper_a, cp.paper_b        # Paper titles
cp.claim_a, cp.claim_b        # The opposing claims
cp.contested_topic            # What they disagree on
cp.severity                   # "low" | "moderate" | "high"
```

---

## Project Structure

```
axiom/
├── __init__.py       — Public exports (LiteratureAgent, Paper, SearchResult, ...)
├── literature.py     — LiteratureAgent: orchestration, Claude prompting, result assembly
├── api_clients.py    — SemanticScholarClient, ArxivClient (HTTP, rate limiting, retry)
├── llm_backends.py   — Pluggable LLM backends (Anthropic, Ollama, OpenAI, OpenRouter)
├── models.py         — Pydantic data models (Paper, SearchResult, ConflictReport, ...)
└── cli.py            — Typer CLI: axiom search / conflicts / gaps / field
```

---

## ⚠️ Semantic Scholar Rate Limits — Read This First

This is the most common cause of errors when running AXIOM.

### The Problem

Semantic Scholar enforces strict rate limits on anonymous requests:

- **Without an API key** — ~1 request/second, shared across *all anonymous users globally*. Under any real load this degrades severely. You will see repeated `429 Too Many Requests` errors even after waiting the full `Retry-After` delay.
- **With an API key** — dedicated quota with significantly higher limits.

In practice, a single `agent.search()` call hits the Semantic Scholar endpoint multiple times. Without a key, the shared anonymous pool is often saturated and the built-in retry logic (6 attempts, 30-second waits) is not enough to recover.

### The Fix — Get a Free API Key (Takes ~2 Minutes)

1. Go to [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
2. Click **"Get API Key"** — fill out a short form (name, email, use case)
3. You receive the key by email, usually within minutes to a few hours
4. Add it to your `.env` file or pass it directly:

```bash
# .env
SEMANTIC_SCHOLAR_API_KEY=your-key-here
```

```python
agent = LiteratureAgent(
    backend="ollama",
    model="mistral",
    ss_api_key="your-key-here"   # or set SEMANTIC_SCHOLAR_API_KEY env var
)
```

With the key, the client automatically:
- Sends the key in the `x-api-key` request header
- Reduces per-request delay from 1.1s to 0.2s

### ArXiv as a Fallback

ArXiv is always free, requires no key, and respects a polite 3-second delay between requests. AXIOM queries both sources by default (`include_arxiv=True`). If Semantic Scholar is throttling, ArXiv results will still be returned and synthesised. For lightweight testing, you can lean on ArXiv alone by keeping the limit low and not worrying about the SS errors:

```python
# ArXiv will still return results even if Semantic Scholar 429s
results = agent.search("your query", include_arxiv=True, limit=10)
```

### Why the Retry Logic Sometimes Still Fails

The retry code in `api_clients.py` reads the `Retry-After` response header (minimum 30s) and retries up to 6 times. But when Semantic Scholar's anonymous pool is heavily loaded, the server continues returning 429 regardless of how long you wait — your wait and their throttle clock are not synchronised. A dedicated API key eliminates this entirely because it has its own quota bucket.

---

## Security Note

Never hardcode API keys in source files. Use environment variables or a `.env` file and ensure `.env` is in your `.gitignore`. Rotate any key that was accidentally committed.

---

## Data Sources

| Source | Coverage | Key Required |
|---|---|---|
| Semantic Scholar | 200M+ papers across all fields | No (but strongly recommended — see above) |
| ArXiv | Preprints in CS, physics, maths, biology | No |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Roadmap

- [ ] `HypothesisAgent` — cross-reference ideas against negative results before running experiments
- [ ] `CollaboratorAgent` — find researchers working on complementary gaps
- [ ] `GrantAgent` — align research framing to funder priorities
- [ ] Persistent knowledge graph per project
- [ ] Weekly monitoring with email digest

---

## License

MIT

---

## Author

**MD Tanvir Anjum** · [contact@voidstudio.tech](mailto:contact@voidstudio.tech) · [Void Studio](https://voidstudio.tech)
