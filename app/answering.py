"""
Evidence Selection, Verification, and Question Answering Module.
Supports local extractive QA via Transformer models and heuristic extractive fallback.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    TRANSFORMERS_QA_AVAILABLE = True
except ImportError:
    TRANSFORMERS_QA_AVAILABLE = False


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex boundary detection."""
    if not text:
        return []
    # Split on sentence ending punctuation followed by whitespace and uppercase or end of string
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text.strip()]


def select_best_evidence_sentence(query: str, passage: str) -> Tuple[str, float]:
    """
    Select the most relevant sentence from a passage given the query.
    Uses token overlap and query coverage.
    """
    sentences = split_into_sentences(passage)
    if not sentences:
        return "", 0.0

    query_tokens = set(re.findall(r"\b[a-z0-9]+\b", query.lower()))
    # Filter out very short common words for lexical overlap
    stopwords = {"what", "is", "the", "how", "why", "when", "where", "who", "which", "are", "do", "does", "did", "can", "in", "of", "to", "a", "an", "for", "on", "with"}
    content_query_tokens = query_tokens - stopwords
    if not content_query_tokens:
        content_query_tokens = query_tokens

    best_sentence = sentences[0]
    best_score = -1.0

    for sentence in sentences:
        sent_tokens = set(re.findall(r"\b[a-z0-9]+\b", sentence.lower()))
        if not sent_tokens:
            continue
        overlap = len(content_query_tokens & sent_tokens)
        score = overlap / (len(content_query_tokens) + 1e-6)

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence, max(0.0, float(best_score))


class ExtractiveQASystem:
    """
    Question Answering engine combining Transformer span extraction
    and sentence-level evidence grounding.
    """

    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        self.model_name = model_name
        self.tokenizer: Any = None
        self.model: Any = None
        self.is_loaded = False
        self.error_message: Optional[str] = None

    def load_model(self) -> bool:
        if not TRANSFORMERS_QA_AVAILABLE:
            self.error_message = "PyTorch or Transformers is not available."
            self.is_loaded = False
            return False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.model.eval()
            self.is_loaded = True
            self.error_message = None
            return True
        except Exception as e:
            self.is_loaded = False
            self.error_message = f"Failed to load QA model '{self.model_name}': {str(e)}"
            return False

    def answer_question(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        confidence_threshold: float = 0.05,
        force_fallback: bool = False
    ) -> Dict[str, Any]:
        """
        Generate answer and select evidence from top retrieved passages.
        
        Args:
            query: The user's natural language question.
            retrieved_passages: Top ranked candidate passages.
            confidence_threshold: Threshold below which abstention is triggered.
            force_fallback: If True, bypasses Transformer model and uses heuristic extractor.
            
        Returns:
            Dict containing answer, evidence, source, verification, and confidence metrics.
        """
        if not retrieved_passages or not query.strip():
            return {
                "answer": "Insufficient evidence in the knowledge base to answer this question reliably.",
                "synthesized_answer": "Insufficient evidence in the knowledge base to answer this question reliably.",
                "citations": [],
                "evidence_sentence": "",
                "source_doc": "",
                "passage_id": "",
                "confidence_score": 0.0,
                "has_sufficient_evidence": False,
                "answering_method": "Abstention (Empty retrieval)",
                "synthesis_method": "Abstention (No grounded context)",
                "verification_status": "No candidate passages available",
                "verification_rule": "Corpus search returned zero candidates."
            }

        top_passage = retrieved_passages[0]
        context_text = top_passage["text"]
        source_doc = top_passage.get("doc_name", "Unknown")
        passage_id = top_passage.get("passage_id", "Unknown")

        # Select evidence sentence
        best_sentence, overlap_score = select_best_evidence_sentence(query, context_text)

        # Compute passage-level content token overlap
        stopwords = {"what", "is", "the", "how", "why", "when", "where", "who", "which", "are", "do", "does", "did", "can", "in", "of", "to", "a", "an", "for", "on", "with"}
        q_content = set(re.findall(r"\b[a-z0-9]+\b", query.lower())) - stopwords
        p_tokens = set(re.findall(r"\b[a-z0-9]+\b", context_text.lower()))
        passage_overlap = (len(q_content & p_tokens) / len(q_content)) if q_content else 0.0

        # Check for completely irrelevant queries across all retrieval modes
        dense_score = top_passage.get("dense_score", None)
        rerank_score = top_passage.get("rerank_score", None)

        low_rerank = (rerank_score is not None and rerank_score < -4.0 and overlap_score <= 0.15)
        no_lexical_and_low_dense = (
            overlap_score <= 0.01
            and passage_overlap <= 0.01
            and (dense_score is None or dense_score < 0.35)
        )

        if low_rerank or no_lexical_and_low_dense:
            return {
                "answer": "Insufficient evidence in the knowledge base to answer this question reliably.",
                "synthesized_answer": "Insufficient evidence in the knowledge base to answer this question reliably.",
                "citations": [],
                "evidence_sentence": best_sentence,
                "source_doc": source_doc,
                "passage_id": passage_id,
                "confidence_score": 0.0,
                "has_sufficient_evidence": False,
                "answering_method": "Abstention (Out-of-domain / Low relevance guardrail)",
                "synthesis_method": "Abstention (Low relevance guardrail)",
                "verification_status": "Unverified (Query terms not grounded in retrieved corpus)",
                "verification_rule": "Zero content-term overlap or negative cross-encoder relevance below threshold."
            }

        synthesizer = GroundedAnswerSynthesizer()

        # Attempt Transformer Extractive QA if enabled
        if not force_fallback and TRANSFORMERS_QA_AVAILABLE:
            if not self.is_loaded:
                self.load_model()

            if self.is_loaded and self.model is not None and self.tokenizer is not None:
                try:
                    inputs = self.tokenizer(
                        query,
                        context_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        return_offsets_mapping=True
                    )
                    offset_mapping = inputs.pop("offset_mapping")[0]

                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    start_logits = outputs.start_logits[0].cpu().numpy()
                    end_logits = outputs.end_logits[0].cpu().numpy()

                    # Find best valid span (max 35 tokens), skipping special token index 0 ([CLS])
                    seq_len = len(start_logits)
                    best_span_score = -1e9
                    best_start = 0
                    best_end = 0

                    for s in range(1, seq_len):
                        if int(offset_mapping[s][0]) == 0 and int(offset_mapping[s][1]) == 0:
                            continue
                        for e in range(s, min(s + 35, seq_len)):
                            if int(offset_mapping[e][0]) == 0 and int(offset_mapping[e][1]) == 0:
                                continue
                            span_score = start_logits[s] + end_logits[e]
                            if span_score > best_span_score:
                                best_span_score = span_score
                                best_start = s
                                best_end = e

                    start_probs = np.exp(start_logits - np.max(start_logits))
                    start_probs /= np.sum(start_probs)
                    end_probs = np.exp(end_logits - np.max(end_logits))
                    end_probs /= np.sum(end_probs)
                    confidence = float(start_probs[best_start] * end_probs[best_end])

                    char_start = int(offset_mapping[best_start][0])
                    char_end = int(offset_mapping[best_end][1])

                    extracted_span = context_text[char_start:char_end].strip()

                    if (
                        not extracted_span
                        or len(extracted_span) < 2
                        or extracted_span == context_text
                        or (extracted_span in {"[CLS]", "[SEP]"} and "format" not in query.lower() and "token" not in query.lower())
                    ):
                        extracted_span = best_sentence

                    if confidence < confidence_threshold and overlap_score < 0.1:
                        return {
                            "answer": "Insufficient evidence in the knowledge base to answer this question reliably.",
                            "synthesized_answer": "Insufficient evidence in the knowledge base to answer this question reliably.",
                            "citations": [],
                            "evidence_sentence": best_sentence,
                            "source_doc": source_doc,
                            "passage_id": passage_id,
                            "confidence_score": round(confidence, 4),
                            "has_sufficient_evidence": False,
                            "answering_method": f"Transformer Extractive QA ({self.model_name}) - Abstention",
                            "synthesis_method": "Abstention (Below confidence threshold)",
                            "verification_status": f"Confidence ({confidence:.4f}) fell below threshold ({confidence_threshold})",
                            "verification_rule": f"Confidence score < {confidence_threshold}."
                        }

                    is_verbatim = extracted_span.lower() in context_text.lower()
                    verification_status = "Verified verbatim in source passage" if is_verbatim else "Derived from context"

                    synth_res = synthesizer.synthesize(
                        query=query,
                        extracted_span=extracted_span,
                        primary_evidence=best_sentence,
                        retrieved_passages=retrieved_passages
                    )

                    return {
                        "answer": extracted_span,
                        "synthesized_answer": synth_res["synthesized_answer"],
                        "citations": synth_res["citations"],
                        "synthesis_method": synth_res["synthesis_method"],
                        "evidence_sentence": best_sentence,
                        "source_doc": source_doc,
                        "passage_id": passage_id,
                        "confidence_score": round(confidence, 4),
                        "has_sufficient_evidence": True,
                        "answering_method": f"Transformer Extractive QA ({self.model_name})",
                        "verification_status": verification_status,
                        "verification_rule": "Token span extracted from DistilBERT logits and verified in passage."
                    }
                except Exception:
                    pass

        # Heuristic Extractive Sentence Fallback
        heuristic_confidence = min(0.95, max(0.20, overlap_score))
        synth_res = synthesizer.synthesize(
            query=query,
            extracted_span=best_sentence,
            primary_evidence=best_sentence,
            retrieved_passages=retrieved_passages
        )
        return {
            "answer": best_sentence,
            "synthesized_answer": synth_res["synthesized_answer"],
            "citations": synth_res["citations"],
            "synthesis_method": synth_res["synthesis_method"],
            "evidence_sentence": best_sentence,
            "source_doc": source_doc,
            "passage_id": passage_id,
            "confidence_score": round(heuristic_confidence, 4),
            "has_sufficient_evidence": True,
            "answering_method": "Extractive Sentence Selection (Lexical Fallback)",
            "verification_status": "Verified: direct sentence extracted from top source passage",
            "verification_rule": "Sentence-level lexical overlap and coverage optimization."
        }


class GroundedAnswerSynthesizer:
    """
    Multi-Passage Extractive & Evidence-Based Answer Synthesizer with Inline Citations.
    Combines the extracted primary answer span with non-redundant supporting
    evidence sentences across top retrieved passages, embedding explicit
    inline citations [1: passage_id]. Operates via deterministic extractive sentence
    aggregation rather than an unconstrained generative LLM.
    """

    def __init__(self, max_supporting_passages: int = 3, min_sentence_overlap: float = 0.20):
        self.max_supporting_passages = max_supporting_passages
        self.min_sentence_overlap = min_sentence_overlap

    def synthesize(
        self,
        query: str,
        extracted_span: str,
        primary_evidence: str,
        retrieved_passages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize a multi-sentence extractive answer with verified inline passage citations.
        """
        if not retrieved_passages or not query.strip():
            return {
                "synthesized_answer": "Insufficient evidence in the knowledge base to answer this question reliably.",
                "citations": [],
                "synthesis_method": "Abstention"
            }

        top_p = retrieved_passages[0]
        top_text = top_p.get("text", "")
        top_pid = top_p.get("passage_id", "Unknown")
        top_doc = top_p.get("doc_name", "Unknown")

        best_top_sent, top_overlap = select_best_evidence_sentence(query, top_text)
        lead_claim = primary_evidence.strip() if primary_evidence.strip() else best_top_sent.strip()
        if not lead_claim:
            lead_claim = extracted_span.strip()

        # Verify that lead_claim or extracted_span is actually present in top_p["text"]
        span_in_top = bool(extracted_span.strip()) and (extracted_span.strip().lower() in top_text.lower())
        claim_in_top = bool(lead_claim) and (lead_claim.lower() in top_text.lower())

        if (not span_in_top and not claim_in_top) or (top_overlap <= 0.01 and not span_in_top):
            return {
                "synthesized_answer": "Insufficient evidence in the knowledge base to answer this question reliably.",
                "citations": [],
                "synthesis_method": "Abstention (Ungrounded or irrelevant context)"
            }

        # Ensure supporting_sentence covers extracted_span when extracted_span comes from a different sentence in top_p
        supporting_sent = lead_claim if claim_in_top else best_top_sent
        if span_in_top and extracted_span.strip().lower() not in supporting_sent.lower():
            for s in split_into_sentences(top_text):
                if extracted_span.strip().lower() in s.lower():
                    supporting_sent = f"{s} {supporting_sent}".strip() if supporting_sent else s
                    break

        if span_in_top and extracted_span.strip().lower() not in lead_claim.lower():
            lead_text = f"{extracted_span.strip().rstrip('.')} — {lead_claim}"
        else:
            lead_text = lead_claim

        citations: List[Dict[str, Any]] = []
        synthesized_parts: List[str] = []
        seen_sentences = set()

        citations.append({
            "citation_num": 1,
            "passage_id": top_pid,
            "doc_name": top_doc,
            "supporting_sentence": supporting_sent,
            "role": "Primary Answer Evidence"
        })
        synthesized_parts.append(f"{lead_text} [1: {top_pid}]")
        seen_sentences.add(lead_claim.lower())
        seen_sentences.add(supporting_sent.lower())

        # Inspect secondary retrieved passages for complementary grounded context
        citation_idx = 2
        for cand in retrieved_passages[1:self.max_supporting_passages]:
            cand_text = cand.get("text", "")
            cand_pid = cand.get("passage_id", "Unknown")
            cand_doc = cand.get("doc_name", "Unknown")

            best_sent, overlap = select_best_evidence_sentence(query, cand_text)
            if (
                best_sent
                and overlap >= self.min_sentence_overlap
                and best_sent.lower() in cand_text.lower()
                and best_sent.lower() not in seen_sentences
            ):
                seen_sentences.add(best_sent.lower())
                citations.append({
                    "citation_num": citation_idx,
                    "passage_id": cand_pid,
                    "doc_name": cand_doc,
                    "supporting_sentence": best_sent,
                    "role": f"Complementary Evidence (overlap={overlap:.2f})"
                })
                formatted_sent = best_sent[0].lower() + best_sent[1:] if len(best_sent) > 1 and not best_sent[:2].isupper() else best_sent
                synthesized_parts.append(f"Furthermore, {formatted_sent} [{citation_idx}: {cand_pid}]")
                citation_idx += 1

        synthesized_answer = " ".join(synthesized_parts)
        return {
            "synthesized_answer": synthesized_answer,
            "citations": citations,
            "synthesis_method": "Multi-Passage Extractive Evidence Synthesis (DistilBERT Span + Cited Sentences)"
        }


