"""
Integration tests for HybridQAEngine covering end-to-end multi-stage pipeline,
fallback behavior, and result serialization.
"""

import os
import json
import pytest
from app.engine import HybridQAEngine


@pytest.fixture(scope="module")
def engine():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_docs = os.path.join(base_dir, "sample_docs")
    eng = HybridQAEngine(sample_docs_dir=sample_docs)
    return eng


def test_system_status(engine):
    status = engine.get_system_status()
    assert status["student_info"]["reg_no"] == "23MID0429"
    assert status["student_info"]["name"] == "Vinay Vishwanathan"
    assert status["document_count"] >= 6
    assert status["passage_count"] >= 6
    assert status["bm25_active"] is True


def test_query_bm25_only(engine):
    res = engine.execute_query("What is BM25?", mode="bm25_only", top_k=3)
    assert res["mode"] == "bm25_only"
    assert len(res["candidates"]) <= 3
    assert res["has_sufficient_evidence"] is True
    assert "bm25" in res["answer"].lower() or "bm25" in res["evidence"].lower()


def test_query_dense_only(engine):
    res = engine.execute_query("What problem does semantic retrieval solve?", mode="dense_only", top_k=3)
    assert len(res["candidates"]) <= 3
    assert res["has_sufficient_evidence"] is True


def test_query_hybrid_rrf(engine):
    res = engine.execute_query("How does Reciprocal Rank Fusion work?", mode="hybrid_rrf", top_k=3)
    assert len(res["candidates"]) <= 3
    assert res["has_sufficient_evidence"] is True
    assert res["candidates"][0]["retrieval_method"] == "hybrid_rrf"


def test_query_hybrid_rerank(engine):
    res = engine.execute_query("Why is a two-stage retrieval pipeline used with cross-encoders?", mode="hybrid_rerank", top_k=3)
    assert len(res["candidates"]) <= 3
    assert res["has_sufficient_evidence"] is True
    assert res["candidates"][0].get("rerank_applied") is True


def test_empty_query(engine):
    res = engine.execute_query("", mode="hybrid_rerank", top_k=3)
    assert res["has_sufficient_evidence"] is False
    assert "Insufficient evidence" in res["answer"]


def test_offline_fallback_mode(engine):
    # Toggle forced offline mode
    engine.set_offline_mode(True)
    res = engine.execute_query("What is BM25?", mode="hybrid_rerank", top_k=3)
    assert res["has_sufficient_evidence"] is True
    # Reset back to online mode
    engine.set_offline_mode(False)


def test_export_serialization(engine):
    res = engine.execute_query("What is Okapi BM25?", mode="hybrid_rerank", top_k=2)
    # JSON export
    json_str = engine.export_query_result_json(res)
    data = json.loads(json_str)
    assert data["student_metadata"]["register_number"] == "23MID0429"
    assert "candidates" in data

    # CSV export
    csv_str = engine.export_candidates_csv(res["candidates"])
    assert "passage_id" in csv_str
    assert "doc_name" in csv_str


def test_query_agentic_hyde_rerank(engine):
    res = engine.execute_query(
        "What is the role of parameter k1 in BM25?",
        mode="agentic_hyde_rerank",
        top_k=3
    )
    assert res["mode"] == "agentic_hyde_rerank"
    assert res["agentic_rewrite_enabled"] is True
    assert res["hyde_enabled"] is True
    assert len(res["react_trace"]) >= 2
    assert len(res["hypothetical_document"]) > 30
    assert res["faiss_used"] is True
    assert len(res["citations"]) >= 1
    assert "[1:" in res["synthesized_answer"]
    ragas = res["ragas_metrics"]
    assert 0.0 <= ragas["faithfulness"] <= 1.0
    assert 0.0 <= ragas["answer_relevance"] <= 1.0
    assert 0.0 <= ragas["context_precision"] <= 1.0
    assert 0.0 <= ragas["context_recall"] <= 1.0


def test_disabled_rewrite_and_hyde_restore_baseline(engine):
    q = "What is the Okapi BM25 algorithm?"
    res_base = engine.execute_query(
        q,
        mode="hybrid_rerank",
        top_k=3,
        use_agentic_rewrite=False,
        use_hyde=False
    )
    assert res_base["agentic_rewrite_enabled"] is False
    assert res_base["hyde_enabled"] is False
    assert res_base["rewritten_query"] == q
    assert res_base["react_trace"] == []
    assert res_base["hypothetical_document"] == ""


def test_unanswerable_query_abstains_in_all_modes(engine):
    unanswerable = "What is the capital of planet Mars?"
    for m in ["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rerank", "agentic_hyde_rerank"]:
        res = engine.execute_query(unanswerable, mode=m, top_k=3)
        assert res["has_sufficient_evidence"] is False, f"Expected abstention in mode {m}"
        assert "Insufficient evidence" in res["answer"]
        assert res["citations"] == []
        assert res["ragas_metrics"]["faithfulness"] == 0.0


