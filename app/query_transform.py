"""
Deterministic ReAct-Style Query Rewriting and
Hypothetical Document Embeddings (HyDE) Module.

Implements:
1. A bounded, deterministic heuristic ReAct-style (Thought -> Action -> Observation)
   workflow for query analysis, acronym/terminology expansion, and two-step retrieval fusion
   (does not require or claim an autonomous generative LLM agent).
2. Deterministic template-based Hypothetical Document Embeddings (HyDE) passage synthesis
   for dense semantic retrieval, bridging query-document phrasing asymmetry.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any, Callable


# General NLP & IR acronym/concept expansions (free of benchmark ground-truth answer leakage)
DOMAIN_EXPANSIONS: Dict[str, List[str]] = {
    "bm25": ["Okapi BM25", "Best Matching 25", "lexical ranking function"],
    "okapi": ["Okapi BM25", "Best Matching 25", "lexical retrieval"],
    "dense": ["Sentence-BERT", "bi-encoder", "continuous vector representations"],
    "semantic": ["contextual embeddings", "bi-encoder", "neural representations"],
    "rrf": ["Reciprocal Rank Fusion", "hybrid rank fusion"],
    "fusion": ["Reciprocal Rank Fusion", "RRF", "hybrid retrieval"],
    "cross": ["cross-encoder", "Transformer reranker", "joint attention"],
    "encoder": ["cross-encoder", "bi-encoder", "Transformer architecture"],
    "reranking": ["second-stage reranker", "cross-encoder", "candidate reordering"],
    "mrr": ["Mean Reciprocal Rank", "ranking evaluation metric"],
    "ndcg": ["Normalized Discounted Cumulative Gain", "graded relevance metric"],
    "extractive": ["extractive question answering", "SQuAD", "context passage"],
    "faiss": ["Facebook AI Similarity Search", "Approximate Nearest Neighbor", "vector index"]
}

_STOPWORDS = {
    "what", "is", "the", "how", "why", "when", "where", "who", "which",
    "are", "do", "does", "did", "can", "in", "of", "to", "a", "an",
    "for", "on", "with", "that", "used", "use", "should"
}


class AgenticQueryRewriter:
    """
    Bounded Heuristic ReAct-Style (Reason + Act) Query Rewriter and Retrieval Refiner.
    Uses a deterministic 2-step workflow (glossary expansion + lexical coverage observation)
    rather than an unconstrained generative LLM agent, guaranteeing reproducible latency
    and zero external API cost.
    """

    def __init__(self, max_steps: int = 2):
        self.max_steps = max(1, min(max_steps, 3))

    def analyze_and_rewrite(self, query: str) -> Dict[str, Any]:
        """
        Execute deterministic ReAct query analysis and terminology expansion.
        Returns rewritten query, sub-queries, expansion keywords, and structured trace.
        """
        clean_q = query.strip()
        if not clean_q:
            return {
                "original_query": query,
                "rewritten_query": "",
                "sub_queries": [],
                "expanded_keywords": [],
                "react_trace": [
                    {
                        "step": 1,
                        "thought": "Input query is empty. Cannot perform query rewriting.",
                        "action": "Abort",
                        "action_input": "",
                        "observation": "Returned empty query."
                    }
                ],
                "is_rewritten": False
            }

        trace: List[Dict[str, Any]] = []
        tokens = re.findall(r"\b[a-z0-9]+\b", clean_q.lower())
        content_tokens = [t for t in tokens if t not in _STOPWORDS]

        # Step 1: Thought - Diagnose query structure & terminology
        detected_concepts: List[str] = []
        expansion_terms: List[str] = []
        for token in content_tokens:
            if token in DOMAIN_EXPANSIONS:
                detected_concepts.append(token)
                for exp in DOMAIN_EXPANSIONS[token]:
                    if exp.lower() not in clean_q.lower() and exp not in expansion_terms:
                        expansion_terms.append(exp)

        # Check for multi-clause / comparative structure
        sub_queries: List[str] = [clean_q]
        if " and " in clean_q.lower() or " that " in clean_q.lower() or " vs " in clean_q.lower():
            parts = re.split(r"\b(?:and|versus|vs\.?)\b", clean_q, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if len(p.strip().split()) >= 3]
            if len(parts) > 1:
                sub_queries = parts

        thought_1 = (
            f"Query contains {len(content_tokens)} content terms ({', '.join(content_tokens[:6])}). "
            f"Matched glossary concepts: {detected_concepts if detected_concepts else 'none'}. "
            f"Checking deterministic terminology expansion."
        )

        # Construct rewritten query preserving original question intent + top 3 expansions
        selected_expansions = expansion_terms[:3]
        if selected_expansions:
            rewritten_query = f"{clean_q} ({', '.join(selected_expansions)})"
        else:
            rewritten_query = clean_q

        trace.append({
            "step": 1,
            "thought": thought_1,
            "action": "ExpandQueryTerms",
            "action_input": clean_q,
            "observation": (
                f"Added {len(selected_expansions)} expansion phrase(s): {selected_expansions}. "
                f"Rewritten query: '{rewritten_query}'."
            )
        })

        return {
            "original_query": clean_q,
            "rewritten_query": rewritten_query,
            "sub_queries": sub_queries,
            "expanded_keywords": selected_expansions if selected_expansions else content_tokens,
            "react_trace": trace,
            "is_rewritten": rewritten_query != clean_q
        }

    def run_agentic_retrieval(
        self,
        query: str,
        retrieve_fn: Callable[[str], List[Dict[str, Any]]],
        min_overlap_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Execute a 2-step deterministic ReAct retrieval workflow:
        1. Retrieves initial candidates with the original query and observes lexical overlap.
        2. If the query was expanded (is_rewritten=True) or initial overlap is below threshold,
           retrieves candidates with the rewritten query and fuses both candidate lists using
           weighted Reciprocal Rank Fusion (0.65 original + 0.35 expanded).
        """
        rewrite_info = self.analyze_and_rewrite(query)
        trace = list(rewrite_info["react_trace"])

        initial_candidates = retrieve_fn(query)

        query_tokens = set(re.findall(r"\b[a-z0-9]+\b", query.lower())) - _STOPWORDS
        top_text = initial_candidates[0]["text"].lower() if initial_candidates else ""
        top_tokens = set(re.findall(r"\b[a-z0-9]+\b", top_text))
        overlap_ratio = len(query_tokens & top_tokens) / max(1, len(query_tokens))
        top_pid = initial_candidates[0].get("passage_id", "None") if initial_candidates else "None"

        rewritten_q = rewrite_info["rewritten_query"]
        is_rewritten = rewrite_info["is_rewritten"]

        if not is_rewritten and overlap_ratio >= min_overlap_threshold and initial_candidates:
            trace.append({
                "step": 2,
                "thought": (
                    f"No glossary expansion needed and top passage '{top_pid}' has sufficient "
                    f"query term overlap ({overlap_ratio:.2f} >= {min_overlap_threshold})."
                ),
                "action": "VerifyAndFinish",
                "action_input": f"top_passage={top_pid}",
                "observation": f"Retained {len(initial_candidates)} initial candidates."
            })
            return {
                "candidates": initial_candidates,
                "rewrite_info": {
                    **rewrite_info,
                    "react_trace": trace,
                    "refinement_triggered": False,
                    "initial_overlap": round(overlap_ratio, 3)
                }
            }

        # Step 2: Execute expanded query search and fuse rankings
        expanded_candidates = retrieve_fn(rewritten_q) if is_rewritten else []

        if not expanded_candidates:
            trace.append({
                "step": 2,
                "thought": (
                    f"Initial top candidate '{top_pid}' had overlap={overlap_ratio:.2f}, "
                    f"but no expansion terms were available."
                ),
                "action": "RetainInitialCandidates",
                "action_input": query,
                "observation": f"Returned {len(initial_candidates)} initial candidates."
            })
            return {
                "candidates": initial_candidates,
                "rewrite_info": {
                    **rewrite_info,
                    "react_trace": trace,
                    "refinement_triggered": False,
                    "initial_overlap": round(overlap_ratio, 3)
                }
            }

        # Weighted rank fusion of initial_candidates (0.65) and expanded_candidates (0.35)
        k_rrf = 60
        fused_scores: Dict[str, float] = {}
        passage_map: Dict[str, Dict[str, Any]] = {}

        for rank, cand in enumerate(initial_candidates, start=1):
            pid = cand["passage_id"]
            passage_map[pid] = dict(cand)
            fused_scores[pid] = fused_scores.get(pid, 0.0) + 0.65 / (k_rrf + rank)

        novel_count = 0
        for rank, cand in enumerate(expanded_candidates, start=1):
            pid = cand["passage_id"]
            if pid not in passage_map:
                passage_map[pid] = dict(cand)
                novel_count += 1
            else:
                # Preserve highest raw bm25_score
                if cand.get("bm25_score", 0.0) > passage_map[pid].get("bm25_score", 0.0):
                    passage_map[pid]["bm25_score"] = cand["bm25_score"]
                    passage_map[pid]["bm25_norm_score"] = cand.get("bm25_norm_score", 0.0)
            fused_scores[pid] = fused_scores.get(pid, 0.0) + 0.35 / (k_rrf + rank)

        sorted_pids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        merged_candidates: List[Dict[str, Any]] = []
        for final_rank, pid in enumerate(sorted_pids, start=1):
            item = passage_map[pid]
            item["bm25_rank"] = final_rank
            merged_candidates.append(item)

        trace.append({
            "step": 2,
            "thought": (
                f"Initial top candidate '{top_pid}' had overlap={overlap_ratio:.2f}. "
                f"Executing expanded query search and fusing initial + expanded rankings."
            ),
            "action": "SearchExpandedQuery",
            "action_input": rewritten_q,
            "observation": (
                f"Retrieved {len(expanded_candidates)} expanded candidates ({novel_count} novel) "
                f"and fused into {len(merged_candidates)} ranked passages."
            )
        })

        return {
            "candidates": merged_candidates,
            "rewrite_info": {
                **rewrite_info,
                "react_trace": trace,
                "refinement_triggered": True,
                "initial_overlap": round(overlap_ratio, 3)
            }
        }


class HyDEGenerator:
    """
    Deterministic Template-Based Hypothetical Document Embeddings (HyDE) Generator.
    Synthesizes a declarative passage framing the user query as a statement so that
    dense bi-encoder search compares document-like phrasing against corpus passages.
    """

    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Weight given to original query embedding vs hypothetical document embedding:
                   v_final = normalize(alpha * v_query + (1 - alpha) * v_hyde_doc)
        """
        self.alpha = max(0.0, min(1.0, alpha))

    def generate_hypothetical_document(
        self,
        query: str,
        prf_passages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a hypothetical declarative passage for a natural language query.
        Uses syntactic question-to-statement transformation and verified pseudo-relevance
        feedback (PRF) snippets only when lexical overlap confirms domain relevance.
        """
        clean_q = query.strip()
        if not clean_q:
            return {
                "query": query,
                "hypothetical_document": "",
                "method": "Empty query",
                "alpha": self.alpha
            }

        tokens = re.findall(r"\b[a-z0-9]+\b", clean_q.lower())
        content_tokens = set(t for t in tokens if t not in _STOPWORDS)

        expansions: List[str] = []
        for t in tokens:
            if t in DOMAIN_EXPANSIONS:
                for exp in DOMAIN_EXPANSIONS[t]:
                    if exp not in expansions:
                        expansions.append(exp)

        # Verify whether PRF top passage actually shares content words with the query
        prf_sentence = ""
        prf_overlap = 0.0
        if prf_passages and len(prf_passages) > 0 and content_tokens:
            top_text = prf_passages[0].get("text", "")
            top_tokens = set(re.findall(r"\b[a-z0-9]+\b", top_text.lower()))
            prf_overlap = len(content_tokens & top_tokens) / len(content_tokens)
            if prf_overlap >= 0.25:
                sents = re.split(r"(?<=[.!?])\s+", top_text.strip())
                if sents and sents[0]:
                    prf_sentence = f" Reference context indicates: {sents[0]}"

        # If query has zero domain expansions and low corpus overlap, avoid injecting NLP boilerplate
        if not expansions and prf_overlap < 0.25:
            q_clean_body = clean_q.rstrip("?.!")
            hypothetical_doc = f"{q_clean_body}."
            return {
                "query": clean_q,
                "hypothetical_document": hypothetical_doc,
                "method": "Minimal Declarative Conversion (Out-of-Domain Guardrail)",
                "alpha": self.alpha,
                "expansion_concepts": []
            }

        # Build declarative technical framing based on question structure
        q_lower = clean_q.lower().rstrip("?.!")
        if q_lower.startswith("what is ") or q_lower.startswith("what are "):
            subject = clean_q[8:].rstrip("?.!").strip()
            lead_sentence = (
                f"In natural language processing and information retrieval, {subject} refers to a specific "
                f"algorithmic mechanism or technical formulation."
            )
        elif q_lower.startswith("how does ") or q_lower.startswith("how do "):
            subject = re.sub(r"^how\s+(?:does|do)\s+", "", clean_q, flags=re.IGNORECASE).rstrip("?.!").strip()
            lead_sentence = (
                f"The process by which {subject} operates involves computing mathematical scores, "
                f"token representations, or ranking transformations over candidate passages."
            )
        elif q_lower.startswith("why is ") or q_lower.startswith("why do "):
            subject = re.sub(r"^why\s+(?:is|do|does)\s+", "", clean_q, flags=re.IGNORECASE).rstrip("?.!").strip()
            lead_sentence = (
                f"The primary architectural reason why {subject} is employed is to balance "
                f"computational efficiency, scalability, and retrieval accuracy."
            )
        else:
            lead_sentence = f"Regarding {clean_q.rstrip('?.!')}, information retrieval systems address this through structured ranking and grounding."

        concept_sentence = ""
        if expansions:
            concept_sentence = f" Associated concepts include {', '.join(expansions[:4])}."

        hypothetical_doc = f"{lead_sentence}{concept_sentence}{prf_sentence}".strip()

        return {
            "query": clean_q,
            "hypothetical_document": hypothetical_doc,
            "method": "Deterministic Template-Based HyDE Synthesis",
            "alpha": self.alpha,
            "expansion_concepts": expansions[:4]
        }

