"""
Information Retrieval Module: BM25 Lexical Search, Dense Semantic Search, and Ranking Fusion.
"""

from __future__ import annotations
import math
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class SimpleBM25Fallback:
    """
    Pure Python BM25 implementation for zero-dependency lexical retrieval fallback.
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 1.0

        # Term frequencies and document frequencies
        self.doc_freqs: Dict[str, int] = {}
        self.tf_list: List[Dict[str, int]] = []

        for doc in corpus_tokens:
            tf: Dict[str, int] = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
            self.tf_list.append(tf)
            for token in tf:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Calculate IDF
        self.idf: Dict[str, float] = {}
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for i, tf in enumerate(self.tf_list):
            score = 0.0
            doc_len = self.doc_lens[i]
            len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl)
            for token in query_tokens:
                if token in tf:
                    t_idf = self.idf.get(token, 0.0)
                    t_tf = tf[token]
                    num = t_tf * (self.k1 + 1.0)
                    denom = t_tf + self.k1 * len_norm
                    score += t_idf * (num / denom)
            scores[i] = score
        return scores


class BM25Retriever:
    """
    Lexical passage retriever using Okapi BM25.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.model: Any = None
        self.tokenized_corpus: List[List[str]] = []
        self.passages: List[Dict[str, Any]] = []
        self.is_indexed = False

    def index(self, passages: List[Dict[str, Any]]) -> None:
        self.passages = passages
        if not passages:
            self.model = None
            self.tokenized_corpus = []
            self.is_indexed = False
            return

        self.tokenized_corpus = [tokenize_text(p["text"]) for p in passages]

        if RANK_BM25_AVAILABLE:
            self.model = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self.model = SimpleBM25Fallback(self.tokenized_corpus, k1=self.k1, b=self.b)

        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.is_indexed or not self.passages:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = np.array(self.model.get_scores(query_tokens), dtype=np.float32)

        # Handle zero or negative scores
        min_score = float(np.min(scores)) if len(scores) > 0 else 0.0
        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0

        # Sort indices descending
        sorted_indices = np.argsort(scores)[::-1]
        top_indices = sorted_indices[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            raw_score = float(scores[idx])
            norm_score = (raw_score - min_score) / (max_score - min_score) if max_score > min_score else (1.0 if raw_score > 0 else 0.0)

            res = dict(self.passages[idx])
            res.update({
                "bm25_score": raw_score,
                "bm25_norm_score": norm_score,
                "bm25_rank": rank,
                "retrieval_method": "bm25"
            })
            results.append(res)

        return results


class DenseRetriever:
    """
    Semantic passage retriever using Sentence-Transformers bi-encoder
    with FAISS Approximate Nearest Neighbor (HNSW / FlatIP) indexing and NumPy fallback.
    Also supports Hypothetical Document Embeddings (HyDE) vector blending.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        faiss_index_type: str = "IndexHNSWFlat"
    ):
        self.model_name = model_name
        self.faiss_index_type = faiss_index_type
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Any = None
        self.faiss_active: bool = False
        self.embedding_dim: int = 384
        self.passages: List[Dict[str, Any]] = []
        self.is_indexed = False
        self.is_loaded = False
        self.error_message: Optional[str] = None

    def load_model(self) -> bool:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.error_message = "sentence-transformers package is not installed."
            self.is_loaded = False
            return False

        try:
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            self.error_message = None
            return True
        except Exception as e:
            self.is_loaded = False
            self.error_message = f"Failed to load SentenceTransformer: {str(e)}"
            return False

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index (IndexHNSWFlat or IndexFlatIP) over L2-normalized float32 embeddings."""
        if not FAISS_AVAILABLE or embeddings is None or len(embeddings) == 0:
            self.faiss_index = None
            self.faiss_active = False
            return

        try:
            vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
            dim = vecs.shape[1]
            self.embedding_dim = dim

            if self.faiss_index_type == "IndexHNSWFlat":
                # Approximate Nearest Neighbor graph index with Inner Product metric
                index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 40
                index.hnsw.efSearch = max(32, len(vecs))
            else:
                # Exact inner-product index
                index = faiss.IndexFlatIP(dim)

            index.add(vecs)
            self.faiss_index = index
            self.faiss_active = True
        except Exception as e:
            self.faiss_index = None
            self.faiss_active = False
            self.error_message = f"FAISS index build warning (using NumPy fallback): {str(e)}"

    def index(self, passages: List[Dict[str, Any]]) -> None:
        self.passages = passages
        if not passages:
            self.embeddings = None
            self.faiss_index = None
            self.faiss_active = False
            self.is_indexed = False
            return

        if not self.is_loaded:
            success = self.load_model()
            if not success:
                self.is_indexed = False
                return

        texts = [p["text"] for p in passages]
        try:
            # Generate L2-normalized embeddings for cosine similarity via inner product
            embs = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            self.embeddings = np.ascontiguousarray(embs, dtype=np.float32)
            self._build_faiss_index(self.embeddings)
            self.is_indexed = True
        except Exception as e:
            self.is_indexed = False
            self.faiss_active = False
            self.error_message = f"Error during passage encoding: {str(e)}"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        hyde_document: Optional[str] = None,
        hyde_alpha: float = 0.6
    ) -> List[Dict[str, Any]]:
        if not self.is_indexed or self.embeddings is None or not self.passages:
            return []

        if not query.strip():
            return []

        try:
            if hyde_document and hyde_document.strip():
                encoded_pair = self.model.encode(
                    [query.strip(), hyde_document.strip()],
                    normalize_embeddings=True
                )
                q_vec = encoded_pair[0]
                h_vec = encoded_pair[1]
                alpha = max(0.0, min(1.0, hyde_alpha))
                blended = alpha * q_vec + (1.0 - alpha) * h_vec
                norm = float(np.linalg.norm(blended))
                query_emb = (blended / norm).astype(np.float32) if norm > 0 else q_vec.astype(np.float32)
                method_tag = "dense_hyde_faiss" if self.faiss_active else "dense_hyde_numpy"
            else:
                query_emb = np.array(
                    self.model.encode([query.strip()], normalize_embeddings=True)[0],
                    dtype=np.float32
                )
                method_tag = "dense_faiss" if self.faiss_active else "dense"

            k_search = min(top_k, len(self.passages))
            if self.faiss_active and self.faiss_index is not None:
                q_mat = np.ascontiguousarray(query_emb.reshape(1, -1), dtype=np.float32)
                faiss_scores, faiss_indices = self.faiss_index.search(q_mat, k_search)
                top_indices = [int(i) for i in faiss_indices[0] if 0 <= int(i) < len(self.passages)]
                top_raw_scores = [float(np.clip(s, -1.0, 1.0)) for s, i in zip(faiss_scores[0], faiss_indices[0]) if 0 <= int(i) < len(self.passages)]
                min_score = float(min(top_raw_scores)) if top_raw_scores else 0.0
                max_score = float(max(top_raw_scores)) if top_raw_scores else 0.0
            else:
                all_scores = np.clip(np.dot(self.embeddings, query_emb), -1.0, 1.0)
                min_score = float(np.min(all_scores)) if len(all_scores) > 0 else 0.0
                max_score = float(np.max(all_scores)) if len(all_scores) > 0 else 0.0
                sorted_indices = np.argsort(all_scores)[::-1]
                top_indices = [int(i) for i in sorted_indices[:k_search]]
                top_raw_scores = [float(all_scores[i]) for i in top_indices]

            results: List[Dict[str, Any]] = []
            for rank, (idx, raw_score) in enumerate(zip(top_indices, top_raw_scores), start=1):
                norm_score = (raw_score - min_score) / (max_score - min_score) if max_score > min_score else (1.0 if raw_score > 0 else 0.0)

                res = dict(self.passages[idx])
                res.update({
                    "dense_score": raw_score,
                    "dense_norm_score": norm_score,
                    "dense_rank": rank,
                    "retrieval_method": method_tag,
                    "faiss_used": self.faiss_active
                })
                results.append(res)

            return results
        except Exception as e:
            self.error_message = f"Error during dense retrieval: {str(e)}"
            return []


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    k_rrf: int = 60,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Candidate Fusion using Reciprocal Rank Fusion (RRF).
    
    Formula:
        RRF_Score(d) = sum_{m in M} 1.0 / (k_rrf + rank_m(d))
        
    Resolves the score incompatibility problem between unbounded BM25 scores
    and bounded [-1.0, 1.0] cosine similarity scores.
    """
    fused_scores: Dict[str, float] = {}
    passage_map: Dict[str, Dict[str, Any]] = {}
    bm25_ranks: Dict[str, int] = {}
    dense_ranks: Dict[str, int] = {}
    bm25_raw: Dict[str, float] = {}
    dense_raw: Dict[str, float] = {}

    # Process BM25 rankings
    for rank, item in enumerate(bm25_results, start=1):
        pid = item["passage_id"]
        bm25_ranks[pid] = rank
        bm25_raw[pid] = item.get("bm25_score", 0.0)
        passage_map[pid] = item
        fused_scores[pid] = fused_scores.get(pid, 0.0) + (1.0 / (k_rrf + rank))

    # Process Dense rankings
    for rank, item in enumerate(dense_results, start=1):
        pid = item["passage_id"]
        dense_ranks[pid] = rank
        dense_raw[pid] = item.get("dense_score", 0.0)
        if pid not in passage_map:
            passage_map[pid] = item
        fused_scores[pid] = fused_scores.get(pid, 0.0) + (1.0 / (k_rrf + rank))

    # Sort passages by RRF score descending
    sorted_pids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    top_pids = sorted_pids[:top_k]

    results: List[Dict[str, Any]] = []
    for final_rank, pid in enumerate(top_pids, start=1):
        item = dict(passage_map[pid])
        item.update({
            "rrf_score": fused_scores[pid],
            "rrf_rank": final_rank,
            "bm25_rank": bm25_ranks.get(pid, None),
            "bm25_score": bm25_raw.get(pid, 0.0),
            "dense_rank": dense_ranks.get(pid, None),
            "dense_score": dense_raw.get(pid, 0.0),
            "retrieval_method": "hybrid_rrf",
            "score_incompatibility_handled": "Reciprocal Rank Fusion (unsupervised rank-based integration)"
        })
        results.append(item)

    return results


def convex_score_fusion(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    alpha: float = 0.5,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Candidate Fusion using Min-Max Normalized Convex Combination.
    
    Formula:
        Score(d) = alpha * norm_BM25(d) + (1 - alpha) * norm_Dense(d)
    """
    fused_scores: Dict[str, float] = {}
    passage_map: Dict[str, Dict[str, Any]] = {}
    bm25_norms: Dict[str, float] = {}
    dense_norms: Dict[str, float] = {}

    for item in bm25_results:
        pid = item["passage_id"]
        bm25_norms[pid] = item.get("bm25_norm_score", 0.0)
        passage_map[pid] = item

    for item in dense_results:
        pid = item["passage_id"]
        dense_norms[pid] = item.get("dense_norm_score", 0.0)
        if pid not in passage_map:
            passage_map[pid] = item

    all_pids = set(passage_map.keys())
    for pid in all_pids:
        b_norm = bm25_norms.get(pid, 0.0)
        d_norm = dense_norms.get(pid, 0.0)
        fused_scores[pid] = alpha * b_norm + (1.0 - alpha) * d_norm

    sorted_pids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    top_pids = sorted_pids[:top_k]

    results: List[Dict[str, Any]] = []
    for rank, pid in enumerate(top_pids, start=1):
        item = dict(passage_map[pid])
        item.update({
            "convex_score": fused_scores[pid],
            "convex_rank": rank,
            "bm25_norm_score": bm25_norms.get(pid, 0.0),
            "dense_norm_score": dense_norms.get(pid, 0.0),
            "retrieval_method": "hybrid_convex"
        })
        results.append(item)

    return results
