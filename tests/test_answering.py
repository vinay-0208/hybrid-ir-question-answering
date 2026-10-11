"""
Unit tests for evidence selection, verification, and question answering.
"""

import pytest
from app.answering import (
    split_into_sentences,
    select_best_evidence_sentence,
    ExtractiveQASystem,
    GroundedAnswerSynthesizer
)


def test_split_into_sentences():
    text = "The Okapi BM25 algorithm is probabilistic. It improves upon TF-IDF! Does it use IDF? Yes."
    sents = split_into_sentences(text)
    assert len(sents) == 4
    assert sents[0] == "The Okapi BM25 algorithm is probabilistic."
    assert split_into_sentences("") == []


def test_select_best_evidence_sentence():
    passage = (
        "Search engines index documents. "
        "BM25 incorporates a non-linear term frequency saturation curve governed by parameter k1. "
        "Deep learning models require GPUs."
    )
    query = "What does parameter k1 control in BM25?"
    best_sent, score = select_best_evidence_sentence(query, passage)
    assert "parameter k1" in best_sent
    assert score > 0.0


def test_qa_abstention_on_empty():
    qa = ExtractiveQASystem()
    res = qa.answer_question("What is NLP?", [])
    assert res["has_sufficient_evidence"] is False
    assert "Insufficient evidence" in res["answer"]
    assert res["citations"] == []


def test_qa_abstention_on_irrelevant():
    qa = ExtractiveQASystem()
    candidates = [{
        "passage_id": "p1",
        "doc_name": "nlp.txt",
        "text": "Natural language processing models analyze human language syntax and semantics.",
        "bm25_score": 0.0,
        "dense_score": 0.05,
        "rerank_score": -9.5
    }]
    # Query completely unrelated to passage
    res = qa.answer_question("What is the temperature of the core of Jupiter?", candidates)
    assert res["has_sufficient_evidence"] is False
    assert "Insufficient evidence" in res["answer"]
    assert res["citations"] == []


def test_qa_heuristic_fallback():
    qa = ExtractiveQASystem()
    candidates = [{
        "passage_id": "p1",
        "doc_name": "bm25.txt",
        "text": "The parameter b controls document length normalization in BM25. It is usually set to 0.75.",
        "bm25_score": 5.0,
        "dense_score": 0.85
    }]
    res = qa.answer_question("What does parameter b control in BM25?", candidates, force_fallback=True)
    assert res["has_sufficient_evidence"] is True
    assert "parameter b" in res["answer"].lower()
    assert "Verified" in res["verification_status"]
    assert "[1: p1]" in res["synthesized_answer"]
    assert len(res["citations"]) == 1


def test_grounded_answer_synthesizer_with_citations():
    synth = GroundedAnswerSynthesizer(max_supporting_passages=3, min_sentence_overlap=0.20)
    candidates = [
        {
            "passage_id": "bm25_p0",
            "doc_name": "02_lexical_search_and_bm25.txt",
            "text": "Okapi BM25 is a probabilistic lexical ranking algorithm that improves upon TF-IDF."
        },
        {
            "passage_id": "bm25_p1",
            "doc_name": "02_lexical_search_and_bm25.txt",
            "text": "The Okapi BM25 algorithm incorporates term frequency saturation k1 and length normalization b."
        },
        {
            "passage_id": "irrel_p2",
            "doc_name": "other.txt",
            "text": "Convolutional neural networks process pixel grids in computer vision."
        }
    ]
    res = synth.synthesize(
        query="What is the Okapi BM25 algorithm?",
        extracted_span="a probabilistic lexical ranking algorithm",
        primary_evidence="Okapi BM25 is a probabilistic lexical ranking algorithm that improves upon TF-IDF.",
        retrieved_passages=candidates
    )
    assert "[1: bm25_p0]" in res["synthesized_answer"]
    assert "[2: bm25_p1]" in res["synthesized_answer"]
    # Irrelevant 3rd passage must not be cited
    assert "irrel_p2" not in res["synthesized_answer"]
    assert len(res["citations"]) == 2
    for cit in res["citations"]:
        matched_p = next(c for c in candidates if c["passage_id"] == cit["passage_id"])
        assert cit["supporting_sentence"].lower() in matched_p["text"].lower()


def test_synthesizer_rejects_ungrounded_or_irrelevant_passages():
    synth = GroundedAnswerSynthesizer()
    irrelevant_candidates = [
        {
            "passage_id": "p0",
            "doc_name": "doc.txt",
            "text": "Okapi BM25 is a lexical retrieval function."
        }
    ]
    # Fabricated span and evidence not present in passage + irrelevant query
    res = synth.synthesize(
        query="What is the capital of planet Mars?",
        extracted_span="Olympus Mons City",
        primary_evidence="Olympus Mons City is the capital of Mars.",
        retrieved_passages=irrelevant_candidates
    )
    assert res["citations"] == []
    assert "Insufficient evidence" in res["synthesized_answer"]


