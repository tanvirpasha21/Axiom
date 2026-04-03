"""
Tests for AXIOM Literature Agent.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from axiom.core.models import Paper, SearchResult, ConflictPair, WhiteSpace


def make_paper(**kwargs) -> Paper:
    defaults = dict(
        title="Test Paper on Fraud Detection",
        authors=["Alice Smith", "Bob Jones"],
        year=2024,
        venue="NeurIPS",
        abstract="We propose a new method for fraud detection using transformers.",
        citation_count=100,
        relevance_score=0.9,
    )
    defaults.update(kwargs)
    return Paper(**defaults)


class TestPaperModel:
    def test_str_representation(self):
        p = make_paper()
        s = str(p)
        assert "Test Paper on Fraud Detection" in s
        assert "Alice Smith" in s

    def test_str_et_al_for_many_authors(self):
        p = make_paper(authors=["A", "B", "C", "D"])
        assert "et al." in str(p)

    def test_defaults(self):
        p = Paper(title="Minimal Paper")
        assert p.authors == []
        assert p.relevance_score == 0.0
        assert p.conflict_flag is False

    def test_relevance_score_bounds(self):
        with pytest.raises(Exception):
            Paper(title="x", relevance_score=1.5)


class TestSearchResult:
    def test_top_papers_sorted_by_relevance(self):
        papers = [
            make_paper(title="Low", relevance_score=0.3),
            make_paper(title="High", relevance_score=0.9),
            make_paper(title="Mid", relevance_score=0.6),
        ]
        result = SearchResult(query="test", papers=papers)
        top = result.top_papers(2)
        assert top[0].title == "High"
        assert top[1].title == "Mid"

    def test_negative_results_filter(self):
        papers = [
            make_paper(title="Positive", is_negative_result=False),
            make_paper(title="Negative", is_negative_result=True),
        ]
        result = SearchResult(query="test", papers=papers)
        negs = result.negative_results()
        assert len(negs) == 1
        assert negs[0].title == "Negative"

    def test_conflicting_papers_filter(self):
        papers = [
            make_paper(title="Clean", conflict_flag=False),
            make_paper(title="Conflict", conflict_flag=True),
        ]
        result = SearchResult(query="test", papers=papers)
        conflicts = result.conflicting_papers()
        assert len(conflicts) == 1
        assert conflicts[0].title == "Conflict"


class TestLiteratureAgentInit:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from axiom.agents.literature import LiteratureAgent
        with pytest.raises(ValueError, match="No Anthropic API key"):
            LiteratureAgent(api_key=None)

    def test_initialises_with_key(self):
        from axiom.agents.literature import LiteratureAgent
        agent = LiteratureAgent(api_key="sk-ant-test-key-123", verbose=False)
        assert agent.model is not None
        agent.close()


class TestJsonParsing:
    def test_clean_json(self):
        from axiom.agents.literature import LiteratureAgent
        raw = '{"summary": "test", "conflicts": []}'
        result = LiteratureAgent._parse_json(raw)
        assert result["summary"] == "test"

    def test_json_in_code_fence(self):
        from axiom.agents.literature import LiteratureAgent
        raw = '```json\n{"summary": "fenced"}\n```'
        result = LiteratureAgent._parse_json(raw)
        assert result["summary"] == "fenced"

    def test_invalid_json_returns_empty(self):
        from axiom.agents.literature import LiteratureAgent
        result = LiteratureAgent._parse_json("not json at all {{")
        assert result == {}
