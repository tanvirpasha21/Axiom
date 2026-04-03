"""
AXIOM CLI — run the literature agent from your terminal.

Usage:
    axiom search "fraud detection transformers"
    axiom conflicts "graph neural networks"
    axiom gaps "real-time fraud detection"
    axiom field "fintech fraud detection"

    # Use OpenRouter
    axiom search "RL for drug discovery" --backend openrouter --model openai/gpt-4o
    OPENROUTER_API_KEY=sk-or-... axiom search "RL for drug discovery" --backend openrouter

    # Other options
    axiom search "RL for drug discovery" --year 2022-2024 --no-arxiv
"""

import os
import typer
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="axiom",
    help="AXIOM — Agentic literature intelligence for researchers.",
    add_completion=False,
)


def _get_agent(**kwargs):
    from axiom.literature import LiteratureAgent

    backend = kwargs.pop("backend", None) or os.environ.get("AXIOM_BACKEND", "anthropic")
    model = kwargs.pop("model", None) or os.environ.get("AXIOM_MODEL")
    base_url = kwargs.pop("base_url", None) or os.environ.get("AXIOM_BASE_URL")

    # Resolve API key: explicit arg → env var for that backend → generic fallback
    api_key = kwargs.pop("api_key", None)
    if not api_key:
        if backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
        elif backend == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif backend == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")

    return LiteratureAgent(
        backend=backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )


_backend_help = "LLM backend: anthropic, openrouter, ollama, openai"
_model_help = (
    "Model name. Examples:\n"
    "  anthropic  → claude-opus-4-5\n"
    "  openrouter → openai/gpt-4o, anthropic/claude-3.5-sonnet, google/gemini-pro-1.5\n"
    "  ollama     → llama2, mistral\n"
    "  openai     → gpt-4o"
)


@app.command()
def search(
    query: str = typer.Argument(..., help="Research question or keywords"),
    year: Optional[str] = typer.Option(None, "--year", help="Year range e.g. 2020-2024"),
    no_arxiv: bool = typer.Option(False, "--no-arxiv", help="Skip ArXiv preprints"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max papers to retrieve"),
    backend: Optional[str] = typer.Option(None, "--backend", help=_backend_help),
    model: Optional[str] = typer.Option(None, "--model", help=_model_help),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="API key. For OpenRouter set OPENROUTER_API_KEY; for Anthropic set ANTHROPIC_API_KEY",
        envvar=["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
    ),
):
    """Search literature and synthesise findings."""
    agent = _get_agent(api_key=api_key, backend=backend, model=model)
    agent.search(query, year_range=year, include_arxiv=not no_arxiv, limit=limit)


@app.command()
def conflicts(
    query: str = typer.Argument(..., help="Research area to analyse for conflicts"),
    backend: Optional[str] = typer.Option(None, "--backend", help=_backend_help),
    model: Optional[str] = typer.Option(None, "--model", help=_model_help),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        envvar=["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
    ),
):
    """Detect conflicting claims across papers in a field."""
    agent = _get_agent(api_key=api_key, backend=backend, model=model)
    agent.find_conflicts(query)


@app.command()
def gaps(
    query: str = typer.Argument(..., help="Research area to scan for white spaces"),
    backend: Optional[str] = typer.Option(None, "--backend", help=_backend_help),
    model: Optional[str] = typer.Option(None, "--model", help=_model_help),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        envvar=["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
    ),
):
    """Surface unaddressed research gaps and white spaces."""
    agent = _get_agent(api_key=api_key, backend=backend, model=model)
    agent.find_white_spaces(query)


@app.command()
def field(
    name: str = typer.Argument(..., help="Research field to summarise"),
    backend: Optional[str] = typer.Option(None, "--backend", help=_backend_help),
    model: Optional[str] = typer.Option(None, "--model", help=_model_help),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        envvar=["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
    ),
):
    """Get a high-level intelligence snapshot of a research field."""
    agent = _get_agent(api_key=api_key, backend=backend, model=model)
    agent.field_summary(name)


if __name__ == "__main__":
    app()
