"""
Unit tests for Agentic Query Rewriting (ReAct workflow) and
Hypothetical Document Embeddings (HyDE) in app/query_transform.py.
"""

import pytest
from app.query_transform import AgenticQueryRewriter, HyDEGenerator


def test_agentic_query_rewriter_basic():
    rewriter = AgenticQueryRewriter(max_steps=2)
    res = rewriter.analyze_and_rewrite("What is the Okapi BM25 algorithm?")
    assert res["original_query"] == "What is the Okapi BM25 algorithm?"
    assert res["is_rewritten"] is True
    assert "BM25" in res["rewritten_query"]
    assert len(res["react_trace"]) >= 1
    assert res["react_trace"][0]["action"] == "ExpandQueryTerms"


def test_agentic_query_rewriter_empty():
    rewriter = AgenticQueryRewriter()
    res = rewriter.analyze_and_rewrite("   ")
    assert res["rewritten_query"] == ""
    assert res["is_rewritten"] is False
    assert res["react_trace"][0]["action"] == "Abort"


def test_agentic_retrieval_workflow():
    rewriter = AgenticQueryRewriter(max_steps=2)

    def mock_retrieve(q: str):
        if "Okapi" in q:
            return [{"passage_id": "p0", "text": "Okapi BM25 is a probabilistic lexical ranking algorithm."}]
        return []

    out = rewriter.run_agentic_retrieval("What is the Okapi BM25 algorithm?", mock_retrieve)
    assert len(out["candidates"]) == 1
    assert out["candidates"][0]["passage_id"] == "p0"
    assert len(out["rewrite_info"]["react_trace"]) == 2


def test_hyde_generator_synthesis():
    hyde = HyDEGenerator(alpha=0.6)
    res = hyde.generate_hypothetical_document("How does Reciprocal Rank Fusion calculate the score?")
    assert res["query"] == "How does Reciprocal Rank Fusion calculate the score?"
    assert len(res["hypothetical_document"]) > 40
    assert "Reciprocal Rank Fusion" in res["hypothetical_document"] or "mechanism" in res["hypothetical_document"]
    assert res["alpha"] == 0.6


def test_hyde_generator_empty():
    hyde = HyDEGenerator()
    res = hyde.generate_hypothetical_document("")
    assert res["hypothetical_document"] == ""
