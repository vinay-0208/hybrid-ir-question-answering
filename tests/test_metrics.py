"""
Unit tests for evaluation metrics: Precision@k, Recall@k, MRR, NDCG@k, Exact Match, Token F1.
Covers edge cases: perfect retrieval, completely irrelevant retrieval, partial retrieval,
duplicate results, empty relevance labels, and ranking-order changes.
"""

import pytest
import math
from app.metrics import (
    normalize_text_for_eval,
    compute_exact_match,
    compute_token_f1,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_ndcg_at_k,
    compute_faithfulness,
    compute_answer_relevance,
    compute_context_precision,
    compute_context_recall,
    compute_ragas_metrics
)


def test_normalize_text_for_eval():
    assert normalize_text_for_eval("The Okapi BM-25, model!") == "okapi bm25 model"
    assert normalize_text_for_eval("   An Artificial Intelligence. ") == "artificial intelligence"
    assert normalize_text_for_eval("") == ""
    assert normalize_text_for_eval(None) == ""


def test_compute_exact_match():
    assert compute_exact_match("The Model", "model") == 1.0
    assert compute_exact_match("Sentence Transformers", "sentence transformers") == 1.0
    assert compute_exact_match("BM25", "TF-IDF") == 0.0
    assert compute_exact_match("", "") == 1.0
    assert compute_exact_match("Answer", "") == 0.0


def test_compute_token_f1():
    # Exact overlap
    f1, p, r = compute_token_f1("natural language processing", "natural language processing")
    assert f1 == 1.0
    assert p == 1.0
    assert r == 1.0

    # Partial overlap
    f1, p, r = compute_token_f1("language processing", "natural language processing")
    assert round(p, 2) == 1.00
    assert round(r, 2) == 0.67
    assert round(f1, 2) == 0.80

    # Zero overlap
    f1, p, r = compute_token_f1("apple orange", "banana grape")
    assert f1 == 0.0
    assert p == 0.0
    assert r == 0.0

    # Empty inputs
    assert compute_token_f1("", "") == (1.0, 1.0, 1.0)
    assert compute_token_f1("word", "") == (0.0, 0.0, 0.0)


def test_perfect_retrieval():
    """All relevant items retrieved in top positions."""
    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
    relevant = {"doc_A", "doc_B"}
    rel_scores = {"doc_A": 1.0, "doc_B": 1.0}
    k = 5

    p = compute_precision_at_k(retrieved, relevant, k=k)
    r = compute_recall_at_k(retrieved, relevant, k=k)
    mrr = compute_reciprocal_rank(retrieved, relevant, k=k)
    ndcg = compute_ndcg_at_k(retrieved, rel_scores, k=k)

    assert p == 2.0 / 5.0
    assert r == 1.0
    assert mrr == 1.0
    assert ndcg == 1.0
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= ndcg <= 1.0


def test_completely_irrelevant_retrieval():
    """No relevant items retrieved."""
    retrieved = ["doc_X", "doc_Y", "doc_Z"]
    relevant = {"doc_A", "doc_B"}
    rel_scores = {"doc_A": 1.0, "doc_B": 1.0}
    k = 5

    assert compute_precision_at_k(retrieved, relevant, k=k) == 0.0
    assert compute_recall_at_k(retrieved, relevant, k=k) == 0.0
    assert compute_reciprocal_rank(retrieved, relevant, k=k) == 0.0
    assert compute_ndcg_at_k(retrieved, rel_scores, k=k) == 0.0


def test_partially_relevant_retrieval():
    """1 out of 2 relevant items retrieved at rank 2."""
    retrieved = ["doc_X", "doc_A", "doc_Y", "doc_Z", "doc_W"]
    relevant = {"doc_A", "doc_B"}
    rel_scores = {"doc_A": 1.0, "doc_B": 1.0}
    k = 5

    p = compute_precision_at_k(retrieved, relevant, k=k)
    r = compute_recall_at_k(retrieved, relevant, k=k)
    mrr = compute_reciprocal_rank(retrieved, relevant, k=k)
    ndcg = compute_ndcg_at_k(retrieved, rel_scores, k=k)

    assert p == 1.0 / 5.0
    assert r == 1.0 / 2.0  # 0.5
    assert mrr == 1.0 / 2.0  # 0.5
    # NDCG must be strictly positive but less than 1.0
    assert 0.0 < ndcg < 1.0


def test_duplicate_retrieved_results():
    """Duplicate IDs in retrieved list must NOT inflate Recall or NDCG beyond 1.0."""
    retrieved = ["doc_A", "doc_A", "doc_A", "doc_B", "doc_C"]
    relevant = {"doc_A"}
    rel_scores = {"doc_A": 1.0}
    k = 5

    p = compute_precision_at_k(retrieved, relevant, k=k)
    r = compute_recall_at_k(retrieved, relevant, k=k)
    mrr = compute_reciprocal_rank(retrieved, relevant, k=k)
    ndcg = compute_ndcg_at_k(retrieved, rel_scores, k=k)

    # doc_A is only 1 unique relevant item
    assert p == 1.0 / 5.0
    assert r == 1.0  # NOT 3.0!
    assert mrr == 1.0
    assert ndcg == 1.0  # NOT > 1.0!
    assert 0.0 <= r <= 1.0
    assert 0.0 <= ndcg <= 1.0


def test_empty_relevance_labels_and_edge_cases():
    """Handles empty ground truth or empty retrieved list gracefully."""
    retrieved = ["doc_A", "doc_B"]
    empty_rel = set()
    empty_scores = {}

    assert compute_precision_at_k(retrieved, empty_rel, k=5) == 0.0
    assert compute_recall_at_k(retrieved, empty_rel, k=5) == 0.0
    assert compute_reciprocal_rank(retrieved, empty_rel, k=5) == 0.0
    assert compute_ndcg_at_k(retrieved, empty_scores, k=5) == 0.0

    # Negative or zero k
    assert compute_precision_at_k(retrieved, {"doc_A"}, k=0) == 0.0
    assert compute_recall_at_k(retrieved, {"doc_A"}, k=-1) == 0.0
    assert compute_ndcg_at_k(retrieved, {"doc_A": 1.0}, k=0) == 0.0


def test_ranking_order_changes():
    """Higher rank position produces higher MRR and NDCG."""
    rel = {"doc_A"}
    scores = {"doc_A": 1.0}

    # Case 1: doc_A at rank 1
    ret_rank1 = ["doc_A", "doc_B", "doc_C"]
    mrr_1 = compute_reciprocal_rank(ret_rank1, rel, k=3)
    ndcg_1 = compute_ndcg_at_k(ret_rank1, scores, k=3)

    # Case 2: doc_A at rank 3
    ret_rank3 = ["doc_B", "doc_C", "doc_A"]
    mrr_3 = compute_reciprocal_rank(ret_rank3, rel, k=3)
    ndcg_3 = compute_ndcg_at_k(ret_rank3, scores, k=3)

    assert mrr_1 > mrr_3
    assert mrr_1 == 1.0
    assert round(mrr_3, 3) == round(1.0 / 3.0, 3)

    assert ndcg_1 > ndcg_3
    assert ndcg_1 == 1.0
    # At rank 3: gain = 1/log2(3+1) = 1/log2(4) = 0.5. IDCG = 1/log2(2) = 1.0. NDCG = 0.5.
    assert round(ndcg_3, 2) == 0.50


def test_ragas_faithfulness_and_relevance():
    contexts = ["Okapi BM25 is a probabilistic ranking function that improves upon TF-IDF."]
    grounded_ans = "Okapi BM25 is a probabilistic ranking function [1: bm25_p0]"
    ungrounded_ans = "Quantum computers use superconducting qubits at absolute zero."
    negated_ans = "Okapi BM25 is not a probabilistic ranking function and never improves upon TF-IDF."
    partially_fabricated_ans = "Okapi BM25 is a probabilistic ranking function invented on Mars using alien lasers."

    f_grounded = compute_faithfulness(grounded_ans, contexts)
    f_ungrounded = compute_faithfulness(ungrounded_ans, contexts)
    f_negated = compute_faithfulness(negated_ans, contexts)
    f_partial = compute_faithfulness(partially_fabricated_ans, contexts)

    assert f_grounded == 1.0
    assert f_ungrounded == 0.0
    assert f_negated < 0.30
    assert 0.30 < f_partial < 0.85

    rel_good = compute_answer_relevance("What is the Okapi BM25 ranking function?", grounded_ans)
    rel_irrel = compute_answer_relevance("What is the Okapi BM25 ranking function?", ungrounded_ans)
    rel_abst = compute_answer_relevance("What is BM25?", "Insufficient evidence in the knowledge base to answer this question reliably.")
    assert 0.5 <= rel_good <= 1.0
    assert rel_irrel == 0.0
    assert rel_abst == 0.0


def test_ragas_context_precision_and_recall():
    retrieved = ["p_irrel", "p_rel1", "p_irrel2", "p_rel2", "p_irrel3"]
    relevant = {"p_rel1", "p_rel2"}
    # Rank 2 is relevant (1/2 = 0.5), Rank 4 is relevant (2/4 = 0.5) -> avg = 0.5
    cp = compute_context_precision(retrieved, relevant, k=5)
    assert round(cp, 2) == 0.50

    contexts = ["Cosine similarity is the geometric metric used for unit vectors in dense search."]
    cr = compute_context_recall("Cosine similarity is the geometric metric.", contexts)
    assert cr == 1.0
    assert compute_context_recall("Quantum entanglement", contexts) == 0.0


def test_compute_ragas_metrics_bundle():
    passages = [
        {"passage_id": "p0", "text": "Reciprocal Rank Fusion calculates score as 1 / (k + rank)."}
    ]
    bundle = compute_ragas_metrics(
        query="How does Reciprocal Rank Fusion calculate score?",
        answer="Reciprocal Rank Fusion calculates score as 1 / (k + rank). [1: p0]",
        retrieved_passages=passages,
        ground_truth_answer="Reciprocal Rank Fusion calculates score as 1 / (k + rank).",
        relevant_ids={"p0"},
        k=5
    )
    assert bundle["faithfulness"] == 1.0
    assert bundle["context_precision"] == 1.0
    assert bundle["context_recall"] == 1.0
    assert 0.0 < bundle["answer_relevance"] <= 1.0


def test_benchmark_relevance_label_consistency():
    """Verify that every ground-truth passage ID in data/evaluation_dataset.json exists in corpus and contains ground_truth_evidence."""
    import os
    import json
    from app.ingestion import DocumentRepository

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = DocumentRepository(default_chunk_size=150, default_overlap=30)
    repo.load_sample_documents(os.path.join(base_dir, "sample_docs"))
    passage_lookup = {p["passage_id"]: p["text"] for p in repo.passages}

    with open(os.path.join(base_dir, "data", "evaluation_dataset.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    assert len(dataset) == 10
    for item in dataset:
        rel_pids = item["relevant_passage_ids"]
        assert len(rel_pids) >= 1
        for pid in rel_pids:
            assert pid in passage_lookup, f"Missing passage ID {pid} for {item['id']}"
            assert item["ground_truth_evidence"] in passage_lookup[pid], f"Evidence mismatch in {pid} for {item['id']}"


