"""
Hybrid IR and QA System Engine.
Coordinates ingestion, indexing, retrieval, fusion, reranking, evidence grounding, and export.
"""

from __future__ import annotations
import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from app.ingestion import DocumentRepository
from app.retrieval import BM25Retriever, DenseRetriever, reciprocal_rank_fusion, convex_score_fusion
from app.reranking import CrossEncoderReranker
from app.answering import ExtractiveQASystem
from app.query_transform import AgenticQueryRewriter, HyDEGenerator
from app.metrics import compute_ragas_metrics


class HybridQAEngine:
    """
    Central orchestration engine for Hybrid Information Retrieval & Question Answering.
    Supports Agentic ReAct query rewriting, HyDE dense retrieval, FAISS HNSW ANN search,
    Reciprocal Rank Fusion, Cross-Encoder reranking, Grounded Synthesis with inline citations,
    and RAGAS evaluation.
    """

    STUDENT_NAME = "Vinay Vishwanathan"
    REGISTER_NUMBER = "23MID0429"
    COURSE_CODE = "CSI4001"
    COURSE_NAME = "Natural Language Processing and Computational Linguistics"
    ASSESSMENT = "Case Study 1"
    QUESTION_NUMBER = 29
    PROJECT_TOPIC = "Information Retrieval and Question Answering Using Transformer-Based Models"

    def __init__(self, sample_docs_dir: Optional[str] = None):
        self.repository = DocumentRepository(default_chunk_size=150, default_overlap=30)
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever(faiss_index_type="IndexHNSWFlat")
        self.reranker = CrossEncoderReranker()
        self.qa_system = ExtractiveQASystem()
        self.query_rewriter = AgenticQueryRewriter(max_steps=2)
        self.hyde_generator = HyDEGenerator(alpha=0.6)

        self.sample_docs_dir = sample_docs_dir
        self.is_indexed = False
        self.force_offline_mode = False

        # Load sample docs if available
        if sample_docs_dir and os.path.exists(sample_docs_dir):
            self.load_sample_documents(sample_docs_dir)

    def load_sample_documents(self, sample_docs_dir: str) -> int:
        """Load built-in sample knowledge base and build indexes."""
        count = self.repository.load_sample_documents(sample_docs_dir)
        self.rebuild_indexes()
        return count

    def rebuild_indexes(self) -> None:
        """Build or refresh both BM25 and Dense FAISS vector indices."""
        passages = self.repository.passages
        if not passages:
            self.bm25_retriever.index([])
            self.dense_retriever.index([])
            self.is_indexed = False
            return

        # Build BM25 index
        self.bm25_retriever.index(passages)

        # Build Dense + FAISS index if not in forced offline mode
        if not self.force_offline_mode:
            self.dense_retriever.index(passages)

        self.is_indexed = True

    def set_offline_mode(self, offline: bool) -> None:
        """Toggle forced offline mode (disables dense retrieval and cross-encoder)."""
        self.force_offline_mode = offline
        if offline:
            self.dense_retriever.is_indexed = False
            self.dense_retriever.faiss_active = False
        else:
            if self.repository.passages:
                self.dense_retriever.index(self.repository.passages)

    def warmup_models(self) -> None:
        """Pre-load Cross-Encoder and Extractive QA models to exclude initialization from benchmark query latency."""
        if not self.force_offline_mode:
            if not self.reranker.is_loaded:
                self.reranker.load_model()
            if not self.qa_system.is_loaded:
                self.qa_system.load_model()

    def get_system_status(self) -> Dict[str, Any]:
        """Inspect and report current system health and model availability."""
        import torch
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

        dense_active = (not self.force_offline_mode) and self.dense_retriever.is_indexed and (self.dense_retriever.model is not None)
        faiss_active = dense_active and self.dense_retriever.faiss_active
        rerank_active = (not self.force_offline_mode) and self.reranker.is_loaded and (self.reranker.model is not None)
        qa_active = (not self.force_offline_mode) and self.qa_system.is_loaded and (self.qa_system.model is not None)

        return {
            "student_info": {
                "name": self.STUDENT_NAME,
                "reg_no": self.REGISTER_NUMBER,
                "course": f"{self.COURSE_CODE} - {self.COURSE_NAME}",
                "assessment": f"{self.ASSESSMENT}, Q{self.QUESTION_NUMBER}",
                "topic": self.PROJECT_TOPIC
            },
            "device": device_name,
            "document_count": self.repository.get_document_count(),
            "passage_count": self.repository.get_passage_count(),
            "is_indexed": self.is_indexed,
            "offline_mode_toggled": self.force_offline_mode,
            "bm25_active": self.bm25_retriever.is_indexed,
            "dense_active": dense_active,
            "dense_model_name": self.dense_retriever.model_name if dense_active else "Offline / Inactive",
            "faiss_active": faiss_active,
            "faiss_index_type": self.dense_retriever.faiss_index_type if faiss_active else "NumPy Dot-Product Fallback",
            "rerank_active": rerank_active,
            "rerank_model_name": self.reranker.model_name if rerank_active else "Offline / Inactive",
            "qa_active": qa_active,
            "qa_model_name": self.qa_system.model_name if qa_active else "Extractive Sentence Selection Fallback",
            "agentic_rewriter_active": True,
            "hyde_active": dense_active,
            "ragas_active": True,
            "document_names": self.repository.get_document_names()
        }

    def execute_query(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        top_k: int = 5,
        candidate_pool_size: int = 15,
        use_agentic_rewrite: bool = False,
        use_hyde: bool = False
    ) -> Dict[str, Any]:
        """
        Execute query pipeline according to specified mode.
        Modes:
          - 'bm25_only': Traditional lexical search
          - 'dense_only': Bi-encoder semantic search (with FAISS HNSW ANN)
          - 'hybrid_rrf': Reciprocal rank fusion of BM25 and dense
          - 'hybrid_rerank': Hybrid RRF candidate generation followed by cross-encoder reranking
          - 'agentic_hyde_rerank': Heuristic ReAct query rewriting + HyDE + FAISS + RRF + Cross-Encoder
        """
        start_time = time.perf_counter()
        latencies: Dict[str, float] = {}

        if mode == "agentic_hyde_rerank":
            use_agentic_rewrite = True
            use_hyde = True

        if not self.is_indexed or not self.repository.passages:
            return {
                "query": query,
                "rewritten_query": query,
                "react_trace": [],
                "agentic_rewrite_enabled": use_agentic_rewrite,
                "hyde_enabled": use_hyde,
                "hypothetical_document": "",
                "mode": mode,
                "candidates": [],
                "answer": "No documents are indexed. Please ingest documents first.",
                "synthesized_answer": "No documents are indexed. Please ingest documents first.",
                "citations": [],
                "evidence": "",
                "source_doc": "",
                "passage_id": "",
                "confidence_score": 0.0,
                "has_sufficient_evidence": False,
                "answering_method": "Abstention",
                "latencies_ms": {"total": 0.0},
                "status_notes": "Repository contains 0 indexed passages."
            }

        # Stage 0: Optional Heuristic ReAct Query Analysis & Rewriting
        t_agent_start = time.perf_counter()
        if use_agentic_rewrite:
            rewrite_info = self.query_rewriter.analyze_and_rewrite(query)
        else:
            rewrite_info = {
                "original_query": query,
                "rewritten_query": query,
                "sub_queries": [query] if query.strip() else [],
                "expanded_keywords": [],
                "react_trace": [],
                "is_rewritten": False
            }
        latencies["query_transform_ms"] = round((time.perf_counter() - t_agent_start) * 1000.0, 2)

        # Stage 1: Retrieval (BM25 + Dense FAISS / HyDE)
        t_ret_start = time.perf_counter()
        bm25_candidates: List[Dict[str, Any]] = []
        dense_candidates: List[Dict[str, Any]] = []

        # Run BM25 (with optional ReAct 2-step retrieval fusion if enabled)
        if mode in ["bm25_only", "hybrid_rrf", "hybrid_rerank", "agentic_hyde_rerank"]:
            k_bm25 = top_k if mode == "bm25_only" else candidate_pool_size
            if use_agentic_rewrite:
                agent_out = self.query_rewriter.run_agentic_retrieval(
                    query=query,
                    retrieve_fn=lambda q_str: self.bm25_retriever.retrieve(q_str, top_k=k_bm25)
                )
                bm25_candidates = agent_out["candidates"][:k_bm25]
                rewrite_info = agent_out["rewrite_info"]
            else:
                bm25_candidates = self.bm25_retriever.retrieve(query, top_k=k_bm25)

        # Generate Hypothetical Document (HyDE) only when enabled
        if use_hyde:
            hyde_info = self.hyde_generator.generate_hypothetical_document(
                query=query,
                prf_passages=bm25_candidates[:1] if bm25_candidates else None
            )
        else:
            hyde_info = {
                "query": query,
                "hypothetical_document": "",
                "method": "Disabled",
                "alpha": self.hyde_generator.alpha,
                "expansion_concepts": []
            }

        # Run Dense Retrieval (via FAISS HNSW / FlatIP)
        dense_can_run = (not self.force_offline_mode) and self.dense_retriever.is_indexed
        if mode in ["dense_only", "hybrid_rrf", "hybrid_rerank", "agentic_hyde_rerank"]:
            if dense_can_run:
                k_dense = top_k if mode == "dense_only" else candidate_pool_size
                hyde_doc_to_use = hyde_info["hypothetical_document"] if use_hyde else None
                dense_query = rewrite_info.get("rewritten_query", query) if (use_agentic_rewrite and mode == "dense_only") else query
                dense_candidates = self.dense_retriever.retrieve(
                    query=dense_query,
                    top_k=k_dense,
                    hyde_document=hyde_doc_to_use,
                    hyde_alpha=hyde_info["alpha"]
                )
            elif mode == "dense_only":
                mode = "bm25_only (dense fallback)"
                bm25_candidates = self.bm25_retriever.retrieve(query, top_k=top_k)

        latencies["retrieval_ms"] = round((time.perf_counter() - t_ret_start) * 1000.0, 2)

        # Stage 2: Fusion (if hybrid)
        t_fuse_start = time.perf_counter()
        candidates: List[Dict[str, Any]] = []

        if mode == "bm25_only" or mode.startswith("bm25_only"):
            candidates = bm25_candidates[:top_k]
        elif mode == "dense_only":
            candidates = dense_candidates[:top_k]
        elif mode in ["hybrid_rrf", "hybrid_rerank", "agentic_hyde_rerank"]:
            if dense_can_run and dense_candidates:
                fused = reciprocal_rank_fusion(
                    bm25_results=bm25_candidates,
                    dense_results=dense_candidates,
                    k_rrf=60,
                    top_k=candidate_pool_size
                )
                candidates = fused
            else:
                candidates = bm25_candidates[:candidate_pool_size]
        latencies["fusion_ms"] = round((time.perf_counter() - t_fuse_start) * 1000.0, 2)

        # Stage 3: Reranking (if enabled)
        t_rerank_start = time.perf_counter()
        rerank_active = False
        rerank_status = "Not requested"

        if mode in ["hybrid_rerank", "agentic_hyde_rerank"]:
            if not self.force_offline_mode:
                rerank_query = rewrite_info.get("rewritten_query", query) if use_agentic_rewrite else query
                reranked_pool, rerank_active, rerank_status = self.reranker.rerank(
                    query=rerank_query,
                    candidates=candidates,
                    top_k=top_k
                )
                candidates = reranked_pool
            else:
                candidates = candidates[:top_k]
                rerank_status = "Offline mode active (Reranker disabled)"
        else:
            candidates = candidates[:top_k]

        latencies["reranking_ms"] = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)

        # Stage 4: Evidence Grounding, Extractive QA, and Citation-Grounded Synthesis
        t_qa_start = time.perf_counter()
        qa_output = self.qa_system.answer_question(
            query=query,
            retrieved_passages=candidates,
            force_fallback=self.force_offline_mode
        )
        latencies["qa_ms"] = round((time.perf_counter() - t_qa_start) * 1000.0, 2)

        # Stage 5: Live RAGAS-Style Proxy Quality Evaluation
        dense_model_ref = self.dense_retriever.model if dense_can_run else None
        ragas_scores = compute_ragas_metrics(
            query=query,
            answer=qa_output.get("synthesized_answer", qa_output["answer"]),
            retrieved_passages=candidates,
            k=top_k,
            dense_model=dense_model_ref
        )

        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        latencies["total_ms"] = total_time_ms

        result = {
            "query": query,
            "rewritten_query": rewrite_info.get("rewritten_query", query),
            "react_trace": rewrite_info.get("react_trace", []),
            "agentic_rewrite_enabled": use_agentic_rewrite,
            "hyde_enabled": use_hyde,
            "hypothetical_document": hyde_info.get("hypothetical_document", ""),
            "faiss_used": self.dense_retriever.faiss_active and dense_can_run,
            "faiss_index_type": self.dense_retriever.faiss_index_type if (self.dense_retriever.faiss_active and dense_can_run) else "NumPy Fallback",
            "mode": mode,
            "top_k": top_k,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "answer": qa_output["answer"],
            "synthesized_answer": qa_output.get("synthesized_answer", qa_output["answer"]),
            "citations": qa_output.get("citations", []),
            "synthesis_method": qa_output.get("synthesis_method", ""),
            "evidence": qa_output["evidence_sentence"],
            "source_doc": qa_output["source_doc"],
            "passage_id": qa_output["passage_id"],
            "confidence_score": qa_output["confidence_score"],
            "has_sufficient_evidence": qa_output["has_sufficient_evidence"],
            "answering_method": qa_output["answering_method"],
            "verification_status": qa_output["verification_status"],
            "verification_rule": qa_output["verification_rule"],
            "ragas_metrics": ragas_scores,
            "latencies_ms": latencies,
            "rerank_status": rerank_status
        }
        return result

    def compare_all_modes(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute query across all 5 modes to enable side-by-side comparison."""
        modes = ["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rerank", "agentic_hyde_rerank"]
        comparisons = {}
        for m in modes:
            comparisons[m] = self.execute_query(query=query, mode=m, top_k=top_k)
        return comparisons

    def export_query_result_json(self, result: Dict[str, Any]) -> str:
        """Serialize complete query execution details to formatted JSON string."""
        clean_result = dict(result)
        clean_result["student_metadata"] = {
            "student_name": self.STUDENT_NAME,
            "register_number": self.REGISTER_NUMBER,
            "course": self.COURSE_CODE,
            "assessment": self.ASSESSMENT,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return json.dumps(clean_result, indent=2, default=str)

    def export_candidates_csv(self, candidates: List[Dict[str, Any]]) -> str:
        """Convert candidate passages and ranking scores to CSV format."""
        if not candidates:
            return ""
        df = pd.DataFrame(candidates)
        # Select key columns
        preferred_cols = [
            "passage_id", "doc_name", "rerank_rank", "rerank_score",
            "rrf_rank", "rrf_score", "bm25_rank", "bm25_score",
            "dense_rank", "dense_score", "text"
        ]
        cols_to_use = [c for c in preferred_cols if c in df.columns]
        for c in df.columns:
            if c not in cols_to_use:
                cols_to_use.append(c)
        return df[cols_to_use].to_csv(index=False)
