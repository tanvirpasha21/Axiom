"""
AXIOM — Agentic Research Intelligence Platform
Literature Agent: search, synthesise, detect conflicts, surface white spaces.

Quick start:
    from axiom import LiteratureAgent
    agent = LiteratureAgent()
    results = agent.search("fraud detection transformer models")
    print(results.summary)
"""

from axiom.literature import LiteratureAgent
from axiom.api_clients import LocalPaperStore
from axiom.models import (
    Paper,
    SearchResult,
    ConflictReport,
    WhiteSpaceReport,
    FieldSummary,
)

__version__ = "0.1.0"
__author__ = "MD Tanvir Anjum"
__email__ = "contact@voidstudio.tech"

__all__ = [
    "LiteratureAgent",
    "LocalPaperStore",
    "Paper",
    "SearchResult",
    "ConflictReport",
    "WhiteSpaceReport",
    "FieldSummary",
]
