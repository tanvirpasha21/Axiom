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
from axiom.warehouse import PaperWarehouse
from axiom.ingestion import PaperIngester
from axiom.models import (
    Paper,
    SearchResult,
    ConflictReport,
    WhiteSpaceReport,
    FieldSummary,
    IngestResult,
)

__version__ = "0.2.0"
__author__ = "MD Tanvir Anjum"
__email__ = "contact@voidstudio.tech"

__all__ = [
    # Agent
    "LiteratureAgent",
    # Storage
    "LocalPaperStore",
    "PaperWarehouse",
    "PaperIngester",
    # Models
    "Paper",
    "SearchResult",
    "ConflictReport",
    "WhiteSpaceReport",
    "FieldSummary",
    "IngestResult",
]
