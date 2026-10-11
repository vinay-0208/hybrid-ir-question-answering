"""
Unit tests for Cross-Encoder reranking module.
"""

import pytest
from app.reranking import CrossEncoderReranker


def test_cross_encoder_initialization():
    reranker = CrossEncoderReranker()
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker.is_loaded is False


def test_cross_encoder_rerank_empty_or_fallback():
    reranker = CrossEncoderReranker()
    # Empty query or candidates
    res, active, msg = reranker.rerank("", [])
    assert res == []

    candidates = [
        {"passage_id": "p1", "text": "BM25 scoring function for text."},
        {"passage_id": "p2", "text": "Recipe for chocolate cake."}
    ]

    # Test real or fallback rerank
    reranked, active, msg = reranker.rerank("BM25 retrieval", candidates, top_k=2)
    assert len(reranked) == 2
    assert "rerank_rank" in reranked[0]
    # Check that rerank rank is assigned
    assert reranked[0]["rerank_rank"] == 1
    assert reranked[1]["rerank_rank"] == 2
