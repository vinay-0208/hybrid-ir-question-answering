"""
Cross-Encoder Reranking Module.
Provides second-stage precision reranking over candidate passages.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker:
    """
    Reranks candidate passages using a pretrained Cross-Encoder model.
    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        self.is_loaded = False
        self.error_message: Optional[str] = None

    def load_model(self) -> bool:
        if not CROSS_ENCODER_AVAILABLE:
            self.error_message = "sentence-transformers package is not installed."
            self.is_loaded = False
            return False

        try:
            self.model = CrossEncoder(self.model_name)
            self.is_loaded = True
            self.error_message = None
            return True
        except Exception as e:
            self.is_loaded = False
            self.error_message = f"Failed to load CrossEncoder model: {str(e)}"
            return False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
        """
        Rerank retrieved candidates based on cross-encoder scoring.
        
        Returns:
            (reranked_candidates, is_active, status_message)
        """
        if not candidates or not query.strip():
            return [], True, "Empty query or candidate set."

        if not self.is_loaded:
            success = self.load_model()
            if not success:
                # Graceful fallback: return candidates unchanged with fallback indicator
                for rank, c in enumerate(candidates, start=1):
                    c["rerank_rank"] = rank
                    c["rerank_score"] = None
                    c["rerank_applied"] = False
                return candidates[:top_k] if top_k else candidates, False, self.error_message

        try:
            pairs = [(query.strip(), c["text"]) for c in candidates]
            scores = self.model.predict(pairs)
            if isinstance(scores, (float, int)):
                scores = np.array([scores])
            else:
                scores = np.array(scores)

            # Sort candidates by reranker score descending
            sorted_indices = np.argsort(scores)[::-1]
            if top_k is not None:
                sorted_indices = sorted_indices[:top_k]

            reranked: List[Dict[str, Any]] = []
            for new_rank, idx in enumerate(sorted_indices, start=1):
                item = dict(candidates[idx])
                original_rank = item.get("rrf_rank") or item.get("bm25_rank") or item.get("dense_rank") or (idx + 1)
                item.update({
                    "rerank_score": float(scores[idx]),
                    "rerank_rank": new_rank,
                    "previous_rank": original_rank,
                    "rank_delta": original_rank - new_rank,
                    "rerank_applied": True
                })
                reranked.append(item)

            return reranked, True, "Successfully reranked with cross-encoder."

        except Exception as e:
            err_msg = f"Reranking error during inference: {str(e)}"
            for rank, c in enumerate(candidates, start=1):
                c["rerank_rank"] = rank
                c["rerank_score"] = None
                c["rerank_applied"] = False
            return candidates[:top_k] if top_k else candidates, False, err_msg
