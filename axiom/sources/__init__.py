"""
AXIOM paper source clients.

All clients are async and implement the same interface:
    async def search(query, limit, year_range) -> list[Paper]
"""
from .arxiv import AsyncArxivClient
from .ieee import IEEEClient

__all__ = ["AsyncArxivClient", "IEEEClient"]
