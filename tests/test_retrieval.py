"""
Unit tests for BM25 lexical retrieval, dense semantic retrieval, and fusion algorithms.
"""

import pytest
import numpy as np
from app.retrieval import (
    tokenize_text,
    BM25Retriever,
    DenseRetriever,
    SimpleBM25Fallback,
    reciprocal_rank_fusion,
    convex_score_fusion,
    FAISS_AVAILABLE
)


def test_tokenize_text():
    tokens = tokenize_text("Okapi BM-25 is a 100% probabilistic model!")
    assert tokens == ["okapi", "bm", "25", "is", "a", "100", "probabilistic", "model"]
    assert tokenize_text("") == []


def test_bm25_retriever():
    passages = [
        {"passage_id": "p0", "doc_name": "d0", "text": "Deep learning architectures and neural networks for NLP."},
        {"passage_id": "p1", "doc_name": "d1", "text": "Okapi BM25 is an effective lexical information retrieval function."},
        {"passage_id": "p2", "doc_name": "d2", "text": "Computer vision models process images and videos."}
    ]

    retriever = BM25Retriever()
    retriever.index(passages)
    assert retriever.is_indexed is True

    # Retrieve lexical query
    results = retriever.retrieve("Okapi BM25 lexical search", top_k=2)
    assert len(results) == 2
    assert results[0]["passage_id"] == "p1"
    assert results[0]["bm25_score"] > 0
    assert "bm25_norm_score" in results[0]

    # Empty query
    assert retriever.retrieve("", top_k=2) == []


def test_simple_bm25_fallback():
    corpus = [
        ["information", "retrieval", "search", "engine"],
        ["transformer", "attention", "mechanism"],
        ["vector", "embeddings", "similarity"]
    ]
    bm25 = SimpleBM25Fallback(corpus)
    scores = bm25.get_scores(["information", "retrieval"])
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_reciprocal_rank_fusion():
    bm25_res = [
        {"passage_id": "doc_A", "text": "A text", "bm25_score": 12.5},
        {"passage_id": "doc_B", "text": "B text", "bm25_score": 8.0},
        {"passage_id": "doc_C", "text": "C text", "bm25_score": 2.1}
    ]

    dense_res = [
        {"passage_id": "doc_B", "text": "B text", "dense_score": 0.88},
        {"passage_id": "doc_A", "text": "A text", "dense_score": 0.72},
        {"passage_id": "doc_D", "text": "D text", "dense_score": 0.45}
    ]

    # For k_rrf = 60:
    # doc_A: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # doc_B: 1/(60+2) + 1/(60+1) = 0.03252
    fused = reciprocal_rank_fusion(bm25_res, dense_res, k_rrf=60, top_k=3)
    assert len(fused) == 3
    top_pids = [f["passage_id"] for f in fused]
    assert "doc_A" in top_pids[:2]
    assert "doc_B" in top_pids[:2]
    assert fused[0]["rrf_score"] > 0
    assert "score_incompatibility_handled" in fused[0]


def test_convex_score_fusion():
    bm25_res = [
        {"passage_id": "p1", "text": "one", "bm25_norm_score": 1.0},
        {"passage_id": "p2", "text": "two", "bm25_norm_score": 0.2}
    ]
    dense_res = [
        {"passage_id": "p1", "text": "one", "dense_norm_score": 0.8},
        {"passage_id": "p2", "text": "two", "dense_norm_score": 0.9}
    ]
    fused = convex_score_fusion(bm25_res, dense_res, alpha=0.5, top_k=2)
    # p1 score: 0.5 * 1.0 + 0.5 * 0.8 = 0.90
    # p2 score: 0.5 * 0.2 + 0.5 * 0.9 = 0.55
    assert fused[0]["passage_id"] == "p1"
    assert round(fused[0]["convex_score"], 2) == 0.90


def test_dense_retriever_faiss_and_hyde():
    passages = [
        {"passage_id": "p0", "doc_name": "d0", "text": "Cosine similarity measures the angle between two L2-normalized dense embedding vectors."},
        {"passage_id": "p1", "doc_name": "d1", "text": "Okapi BM25 uses lexical term frequency and inverse document frequency."}
    ]
    dense = DenseRetriever(faiss_index_type="IndexHNSWFlat")
    dense.index(passages)
    assert dense.is_indexed is True
    assert dense.faiss_active == FAISS_AVAILABLE

    # Standard FAISS dense retrieval
    res_std = dense.retrieve("What geometric metric compares dense vector embeddings?", top_k=2)
    assert len(res_std) == 2
    assert res_std[0]["passage_id"] == "p0"
    assert res_std[0]["faiss_used"] == FAISS_AVAILABLE

    # HyDE-enhanced dense retrieval
    res_hyde = dense.retrieve(
        "What geometric metric compares dense vector embeddings?",
        top_k=2,
        hyde_document="In dense retrieval, cosine similarity is the geometric metric used to compare unit vectors.",
        hyde_alpha=0.5
    )
    assert len(res_hyde) == 2
    assert res_hyde[0]["passage_id"] == "p0"
    assert "hyde" in res_hyde[0]["retrieval_method"]


def test_faiss_vs_numpy_retrieval_consistency():
    """Verify that FAISS IndexHNSWFlat and IndexFlatIP match exact NumPy dot-product rankings and L2 norms."""
    passages = [
        {"passage_id": "p0", "doc_name": "d0", "text": "Okapi BM25 is a probabilistic lexical ranking model using term frequency saturation."},
        {"passage_id": "p1", "doc_name": "d1", "text": "Dense semantic retrieval encodes text into continuous vector spaces using bi-encoders."},
        {"passage_id": "p2", "doc_name": "d2", "text": "Reciprocal Rank Fusion combines lexical and semantic rankings via 1 / (k + rank)."},
        {"passage_id": "p3", "doc_name": "d3", "text": "Cross-encoders apply joint self-attention across concatenated query and passage tokens."}
    ]
    dense = DenseRetriever(faiss_index_type="IndexHNSWFlat")
    dense.index(passages)
    assert dense.embeddings is not None
    assert dense.embeddings.shape == (4, 384)

    # Check unit L2 normalization of stored embeddings
    norms = np.linalg.norm(dense.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    query = "How does Reciprocal Rank Fusion combine rankings?"
    dense.faiss_active = True
    res_faiss = dense.retrieve(query, top_k=4)

    dense.faiss_active = False
    res_numpy = dense.retrieve(query, top_k=4)
    dense.faiss_active = FAISS_AVAILABLE

    assert [r["passage_id"] for r in res_faiss] == [r["passage_id"] for r in res_numpy]
    for r_f, r_n in zip(res_faiss, res_numpy):
        assert abs(r_f["dense_score"] - r_n["dense_score"]) < 1e-5


