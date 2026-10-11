"""
Evaluation Metrics Module for Information Retrieval and Question Answering.
Implements mathematically rigorous Recall@k, Precision@k, MRR, NDCG@k, Exact Match, and Token-level F1.
All metrics are strictly bounded in the range [0.0, 1.0].
"""

from __future__ import annotations
import math
import re
import string
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd


def normalize_text_for_eval(text: str) -> str:
    """Normalize text for answer evaluation (lowercase, remove punctuation and articles)."""
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # Remove English articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Normalize whitespace
    return " ".join(text.split())


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute binary Exact Match (EM) score between prediction and ground truth.
    Returns 1.0 if normalized strings are identical, else 0.0.
    """
    norm_pred = normalize_text_for_eval(prediction)
    norm_gt = normalize_text_for_eval(ground_truth)
    if not norm_pred and not norm_gt:
        return 1.0
    return 1.0 if norm_pred == norm_gt else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Compute token-level Precision, Recall, and F1 score.
    Returns (f1, precision, recall), all bounded in [0.0, 1.0].
    """
    norm_pred = normalize_text_for_eval(prediction)
    norm_gt = normalize_text_for_eval(ground_truth)

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()

    if not pred_tokens and not gt_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0, 0.0, 0.0

    common_tokens = set(pred_tokens) & set(gt_tokens)
    if not common_tokens:
        return 0.0, 0.0, 0.0

    # Count overlaps using token multiset frequencies
    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gt_counts: Dict[str, int] = {}
    for t in gt_tokens:
        gt_counts[t] = gt_counts.get(t, 0) + 1

    overlap_count = sum(min(pred_counts[t], gt_counts[t]) for t in common_tokens)

    precision = overlap_count / len(pred_tokens)
    recall = overlap_count / len(gt_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return (
        min(1.0, max(0.0, float(f1))),
        min(1.0, max(0.0, float(precision))),
        min(1.0, max(0.0, float(recall)))
    )


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Precision@k.
    Formula:
        Precision@k = |Unique relevant IDs in top-k| / k
    Handles duplicate retrieved IDs consistently by counting unique relevant matches.
    Strictly bounded: 0.0 <= Precision@k <= 1.0.
    """
    if k <= 0 or not relevant_ids or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    # Unique relevant IDs retrieved within the top-k window
    unique_relevant_retrieved = set(top_k) & set(relevant_ids)
    precision = len(unique_relevant_retrieved) / k
    return min(1.0, max(0.0, float(precision)))


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Recall@k.
    Formula:
        Recall@k = |Unique relevant IDs in top-k| / |Total relevant IDs|
    Strictly bounded: 0.0 <= Recall@k <= 1.0.
    """
    if k <= 0 or not relevant_ids or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    unique_relevant_retrieved = set(top_k) & set(relevant_ids)
    recall = len(unique_relevant_retrieved) / len(relevant_ids)
    return min(1.0, max(0.0, float(recall)))


def compute_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str], k: Optional[int] = None) -> float:
    """
    Compute Reciprocal Rank (RR).
    Finds 1-based rank position of the first relevant item retrieved.
    Formula:
        RR = 1.0 / rank (if rank <= k), else 0.0.
    Strictly bounded: 0.0 <= RR <= 1.0.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    items = retrieved_ids[:k] if k is not None else retrieved_ids
    for rank, item in enumerate(items, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k).
    Formula:
        DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
        IDCG@k = Ideal DCG computed over sorted true relevance scores.
        NDCG@k = DCG@k / IDCG@k
        
    Handles duplicate retrieved IDs by ensuring an already-rewarded item does not receive duplicate gain.
    Strictly bounded: 0.0 <= NDCG@k <= 1.0.
    """
    if k <= 0 or not relevance_scores or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    # Calculate Discounted Cumulative Gain (DCG@k)
    dcg = 0.0
    seen_ids: Set[str] = set()

    for i, item in enumerate(top_k, start=1):
        if item in seen_ids:
            # Duplicate item: already consumed gain at a higher rank
            continue
        seen_ids.add(item)
        rel = relevance_scores.get(item, 0.0)
        if rel > 0.0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 1)

    # Calculate Ideal DCG (IDCG@k)
    ideal_rels = sorted([s for s in relevance_scores.values() if s > 0.0], reverse=True)[:k]
    if not ideal_rels:
        return 0.0

    idcg = 0.0
    for j, rel in enumerate(ideal_rels, start=1):
        idcg += (math.pow(2.0, rel) - 1.0) / math.log2(j + 1)

    if idcg <= 0.0:
        return 0.0

    ndcg = dcg / idcg
    return min(1.0, max(0.0, float(ndcg)))


# =========================================================================
# RAGAS-INSPIRED HEURISTIC PROXY METRICS
# (Faithfulness, Answer Relevance, Context Precision, Context Recall)
# Note: These are deterministic local lexical/bi-encoder proxy metrics
# bounded in [0.0, 1.0], not external LLM-as-a-judge API evaluations.
# =========================================================================

_EVAL_STOPWORDS = {
    "what", "is", "the", "how", "why", "when", "where", "who", "which",
    "are", "do", "does", "did", "can", "in", "of", "to", "a", "an",
    "for", "on", "with", "that", "this", "it", "as", "by", "from", "or",
    "and", "be", "Furthermore", "furthermore"
}

_NEGATION_TOKENS = {"not", "never", "no", "without", "cannot", "neither", "none"}


def _strip_citations(text: str) -> str:
    """Remove bracketed inline citation markers like [1: doc_p0] from synthesized answers."""
    if not text:
        return ""
    return re.sub(r"\[\d+:\s*[^\]]+\]", "", text).strip()


def compute_faithfulness(answer: str, retrieved_contexts: List[str]) -> float:
    """
    RAGAS-inspired heuristic Faithfulness (Groundedness) proxy in [0.0, 1.0].
    Measures whether statements and content tokens in the answer are supported by
    the retrieved context passages, penalizing ungrounded tokens and negation inversions.
    Note: Because extractive QA and extractive sentence synthesis copy text directly
    from the retrieved passages, verbatim extractive outputs score 1.0 against their
    retrieved context even if the retrieved passage is not the ground-truth passage.
    """
    clean_ans = _strip_citations(answer)
    if not clean_ans or not retrieved_contexts:
        return 0.0

    if "insufficient evidence in the knowledge base" in clean_ans.lower():
        return 0.0

    combined_context = " ".join(retrieved_contexts).lower()
    context_sents: List[str] = []
    for ctx in retrieved_contexts:
        for s in re.split(r"(?<=[.!?])\s+", ctx.strip()):
            if s.strip():
                context_sents.append(s.strip().lower())
    if not context_sents:
        context_sents = [combined_context]

    context_tokens = set(re.findall(r"\b[a-z0-9]+\b", combined_context)) - {w.lower() for w in _EVAL_STOPWORDS}

    # Split answer into statements
    statements = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\s+—\s+", clean_ans) if s.strip()]
    if not statements:
        statements = [clean_ans]

    stmt_scores: List[float] = []

    for stmt in statements:
        stmt_clean = re.sub(r"^furthermore,\s*", "", stmt.strip(), flags=re.IGNORECASE)
        stmt_all_words = set(re.findall(r"\b[a-z0-9]+\b", stmt_clean.lower()))
        stmt_tokens = stmt_all_words - {w.lower() for w in _EVAL_STOPWORDS}
        if not stmt_tokens:
            continue

        if stmt_clean.lower() in combined_context:
            stmt_scores.append(1.0)
            continue

        # Compare against each context sentence for best local support and check negation polarity
        best_sent_support = 0.0
        stmt_has_neg = bool(stmt_all_words & _NEGATION_TOKENS)
        for c_sent in context_sents:
            c_all_words = set(re.findall(r"\b[a-z0-9]+\b", c_sent))
            c_tokens = c_all_words - {w.lower() for w in _EVAL_STOPWORDS}
            if not c_tokens:
                continue
            overlap = len(stmt_tokens & c_tokens) / len(stmt_tokens)
            c_has_neg = bool(c_all_words & _NEGATION_TOKENS)
            if stmt_has_neg != c_has_neg and overlap > 0.3:
                # Penalize negation polarity inversion on overlapping claims
                overlap *= 0.25
            if overlap > best_sent_support:
                best_sent_support = overlap

        if best_sent_support < 0.35:
            stmt_scores.append(0.0)
        else:
            stmt_scores.append(min(1.0, best_sent_support))

    if not stmt_scores:
        return 0.0

    stmt_score = sum(stmt_scores) / len(stmt_scores)
    all_ans_tokens = set(re.findall(r"\b[a-z0-9]+\b", clean_ans.lower())) - {w.lower() for w in _EVAL_STOPWORDS}
    token_grounding = (len(all_ans_tokens & context_tokens) / len(all_ans_tokens)) if all_ans_tokens else 0.0

    # If negation inversion penalized statement score, reflect it in final score
    faithfulness = min(stmt_score, 0.5 * stmt_score + 0.5 * token_grounding)
    return min(1.0, max(0.0, float(faithfulness)))


def compute_answer_relevance(query: str, answer: str, dense_model: Any = None) -> float:
    """
    RAGAS-inspired heuristic Answer Relevance proxy in [0.0, 1.0].
    Evaluates bi-encoder cosine similarity (0.65 weight) and query content-term
    coverage (0.35 weight) between the question and answer without an external LLM.
    """
    clean_ans = _strip_citations(answer)
    clean_q = query.strip()
    if not clean_ans or not clean_q:
        return 0.0

    if "insufficient evidence in the knowledge base" in clean_ans.lower():
        return 0.0

    q_tokens = set(re.findall(r"\b[a-z0-9]+\b", clean_q.lower())) - {w.lower() for w in _EVAL_STOPWORDS}
    a_tokens = set(re.findall(r"\b[a-z0-9]+\b", clean_ans.lower())) - {w.lower() for w in _EVAL_STOPWORDS}

    lexical_coverage = (len(q_tokens & a_tokens) / len(q_tokens)) if q_tokens else 0.5

    if dense_model is not None:
        try:
            import numpy as np
            vecs = dense_model.encode([clean_q, clean_ans], normalize_embeddings=True)
            cos_sim = float(np.clip(np.dot(vecs[0], vecs[1]), 0.0, 1.0))
            score = 0.65 * cos_sim + 0.35 * lexical_coverage
            return min(1.0, max(0.0, float(score)))
        except Exception:
            pass

    return min(1.0, max(0.0, float(lexical_coverage)))


def compute_context_precision(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    RAGAS-style Context Precision@k (Average Precision of relevant contexts in top-k) in [0.0, 1.0].
    Measures whether relevant context passages are ranked at the top positions of the retrieved list:
        ContextPrecision@k = (sum_{r=1}^k Precision@r * rel_r) / |Unique relevant IDs in top-k|
    Note: When each query has a single ground-truth relevant passage (|relevant_ids| = 1),
    ContextPrecision@k equals Reciprocal Rank (1/r).
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    seen_relevant: Set[str] = set()
    precision_sum = 0.0

    for rank, pid in enumerate(top_k, start=1):
        if pid in relevant_ids and pid not in seen_relevant:
            seen_relevant.add(pid)
            precision_at_r = len(seen_relevant) / rank
            precision_sum += precision_at_r

    if not seen_relevant:
        return 0.0

    context_prec = precision_sum / len(seen_relevant)
    return min(1.0, max(0.0, float(context_prec)))


def compute_context_recall(ground_truth_answer: str, retrieved_contexts: List[str]) -> float:
    """
    RAGAS-inspired heuristic Context Recall proxy in [0.0, 1.0].
    Measures the proportion of ground-truth reference content tokens present
    in the retrieved context passages.
    """
    if not ground_truth_answer or not retrieved_contexts:
        return 0.0

    combined_context = " ".join(retrieved_contexts).lower()
    context_tokens = set(re.findall(r"\b[a-z0-9]+\b", combined_context)) - {w.lower() for w in _EVAL_STOPWORDS}
    gt_tokens = set(re.findall(r"\b[a-z0-9]+\b", ground_truth_answer.lower())) - {w.lower() for w in _EVAL_STOPWORDS}

    if not gt_tokens:
        return 1.0 if ground_truth_answer.lower() in combined_context else 0.0

    recall = len(gt_tokens & context_tokens) / len(gt_tokens)
    return min(1.0, max(0.0, float(recall)))


def compute_ragas_metrics(
    query: str,
    answer: str,
    retrieved_passages: List[Dict[str, Any]],
    ground_truth_answer: Optional[str] = None,
    relevant_ids: Optional[Set[str]] = None,
    k: int = 5,
    dense_model: Any = None
) -> Dict[str, float]:
    """
    Compute RAGAS-inspired heuristic evaluation profile for a single query execution.
    All metrics are strictly bounded in [0.0, 1.0].
    """
    contexts = [p.get("text", "") for p in retrieved_passages[:k] if p.get("text")]
    retrieved_ids = [p.get("passage_id", "") for p in retrieved_passages[:k]]

    faithfulness = compute_faithfulness(answer, contexts)
    ans_relevance = compute_answer_relevance(query, answer, dense_model=dense_model)

    if relevant_ids:
        ctx_precision = compute_context_precision(retrieved_ids, relevant_ids, k)
    else:
        # Unsupervised context-query lexical overlap proxy for live ad-hoc queries
        if contexts:
            q_tokens = set(re.findall(r"\b[a-z0-9]+\b", query.lower())) - {w.lower() for w in _EVAL_STOPWORDS}
            top_ctx_tokens = set(re.findall(r"\b[a-z0-9]+\b", contexts[0].lower())) - {w.lower() for w in _EVAL_STOPWORDS}
            ctx_precision = (len(q_tokens & top_ctx_tokens) / len(q_tokens)) if q_tokens else 0.0
        else:
            ctx_precision = 0.0

    if ground_truth_answer:
        ctx_recall = compute_context_recall(ground_truth_answer, contexts)
    else:
        # Context utilization against generated answer when ground truth is not provided
        ctx_recall = faithfulness

    return {
        "faithfulness": round(min(1.0, max(0.0, float(faithfulness))), 3),
        "answer_relevance": round(min(1.0, max(0.0, float(ans_relevance))), 3),
        "context_precision": round(min(1.0, max(0.0, float(ctx_precision))), 3),
        "context_recall": round(min(1.0, max(0.0, float(ctx_recall))), 3)
    }


class EvaluationRunner:
    """
    Runs automated evaluation across multiple retrieval modes on a labelled benchmark dataset.
    Evaluates BM25-only, Dense-only, Hybrid (RRF), Hybrid + Cross-Encoder reranking,
    and Heuristic ReAct + HyDE + Reranking under identical queries, candidate cutoffs,
    and relevance ground truth.
    Pre-warms Transformer models prior to timing so model initialization is excluded.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def run_benchmark(
        self,
        dataset: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate all retrieval modes on the benchmark dataset.
        """
        modes = ["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rerank", "agentic_hyde_rerank"]
        mode_labels = {
            "bm25_only": "1. BM25 Only",
            "dense_only": "2. Dense Only (FAISS)",
            "hybrid_rrf": "3. Hybrid (RRF)",
            "hybrid_rerank": "4. Hybrid + Cross-Encoder",
            "agentic_hyde_rerank": "5. Heuristic ReAct + HyDE + Rerank"
        }

        # Pre-load QA and Reranker models before benchmark loop so cold-start load time is excluded
        if hasattr(self.engine, "warmup_models"):
            self.engine.warmup_models()

        dense_model = None
        if hasattr(self.engine, "dense_retriever") and self.engine.dense_retriever.is_loaded:
            dense_model = self.engine.dense_retriever.model

        results_by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
        summary_rows = []

        for m in modes:
            precisions: List[float] = []
            recalls: List[float] = []
            mrrs: List[float] = []
            ndcgs: List[float] = []
            ems: List[float] = []
            f1s: List[float] = []
            faithfulness_list: List[float] = []
            ans_rel_list: List[float] = []
            ctx_prec_list: List[float] = []
            ctx_rec_list: List[float] = []
            latencies: List[float] = []

            for item in dataset:
                q = item["question"]
                gt_ans = item.get("ground_truth_answer", "")

                # Determine target evaluation IDs (passage-level preferred, fallback to doc-level)
                if "relevant_passage_ids" in item and item["relevant_passage_ids"]:
                    rel_ids = set(item["relevant_passage_ids"])
                    id_key = "passage_id"
                else:
                    rel_ids = {item.get("relevant_doc", "")} if item.get("relevant_doc") else set()
                    id_key = "doc_name"

                rel_scores = {rid: 1.0 for rid in rel_ids}

                t0 = time.perf_counter()
                query_res = self.engine.execute_query(query=q, mode=m, top_k=top_k)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                candidates = query_res.get("candidates", [])
                retrieved_ids = [c.get(id_key, "") for c in candidates]
                pred_ans = query_res.get("answer", "")
                synth_ans = query_res.get("synthesized_answer", pred_ans)

                p_k = compute_precision_at_k(retrieved_ids, rel_ids, top_k)
                r_k = compute_recall_at_k(retrieved_ids, rel_ids, top_k)
                rr = compute_reciprocal_rank(retrieved_ids, rel_ids, top_k)
                ndcg = compute_ndcg_at_k(retrieved_ids, rel_scores, top_k)
                em = compute_exact_match(pred_ans, gt_ans)
                f1, _, _ = compute_token_f1(pred_ans, gt_ans)

                ragas = compute_ragas_metrics(
                    query=q,
                    answer=synth_ans,
                    retrieved_passages=candidates,
                    ground_truth_answer=gt_ans,
                    relevant_ids=rel_ids,
                    k=top_k,
                    dense_model=dense_model
                )

                precisions.append(p_k)
                recalls.append(r_k)
                mrrs.append(rr)
                ndcgs.append(ndcg)
                ems.append(em)
                f1s.append(f1)
                faithfulness_list.append(ragas["faithfulness"])
                ans_rel_list.append(ragas["answer_relevance"])
                ctx_prec_list.append(ragas["context_precision"])
                ctx_rec_list.append(ragas["context_recall"])
                latencies.append(latency_ms)

                results_by_mode[m].append({
                    "question_id": item.get("id", ""),
                    "question": q,
                    "target_ids": sorted(list(rel_ids)),
                    "retrieved_top1": retrieved_ids[0] if retrieved_ids else "None",
                    "precision_at_k": round(p_k, 3),
                    "recall_at_k": round(r_k, 3),
                    "reciprocal_rank": round(rr, 3),
                    "ndcg_at_k": round(ndcg, 3),
                    "em": em,
                    "f1": round(f1, 3),
                    "faithfulness": ragas["faithfulness"],
                    "answer_relevance": ragas["answer_relevance"],
                    "context_precision": ragas["context_precision"],
                    "context_recall": ragas["context_recall"],
                    "latency_ms": round(latency_ms, 2)
                })

            summary_rows.append({
                "Retrieval Mode": mode_labels[m],
                f"Precision@{top_k}": round(float(pd.Series(precisions).mean()), 3),
                f"Recall@{top_k}": round(float(pd.Series(recalls).mean()), 3),
                "MRR": round(float(pd.Series(mrrs).mean()), 3),
                f"NDCG@{top_k}": round(float(pd.Series(ndcgs).mean()), 3),
                "Exact Match": round(float(pd.Series(ems).mean()), 3),
                "Token F1": round(float(pd.Series(f1s).mean()), 3),
                "Faithfulness": round(float(pd.Series(faithfulness_list).mean()), 3),
                "Answer Relevance": round(float(pd.Series(ans_rel_list).mean()), 3),
                "Context Precision": round(float(pd.Series(ctx_prec_list).mean()), 3),
                "Context Recall": round(float(pd.Series(ctx_rec_list).mean()), 3),
                "Avg Latency (ms)": round(float(pd.Series(latencies).mean()), 2)
            })

        df_summary = pd.DataFrame(summary_rows)
        return {
            "summary_df": df_summary,
            "detailed_results": results_by_mode
        }

