"""
Document Ingestion and Passage Segmentation Module.
Handles document extraction, cleaning, chunking, and repository management.
"""

from __future__ import annotations
import os
import re
import io
from typing import Dict, List, Optional, Tuple, Any

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def clean_text(text: str) -> str:
    """Normalize whitespace and control characters in raw extracted text."""
    if not text:
        return ""
    # Replace non-breaking spaces and excessive vertical/horizontal whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> Tuple[str, Optional[str]]:
    """
    Extract text content from uploaded file bytes (txt or pdf).
    Returns (extracted_text, error_message).
    """
    if not file_bytes:
        return "", "File is empty (0 bytes)."

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        # Attempt UTF-8 decoding first, then fallback to latin-1
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                return "", f"Text decoding error: {str(e)}"
        cleaned = clean_text(text)
        if not cleaned:
            return "", "File contains only blank or unprintable characters."
        return cleaned, None

    elif ext == ".pdf":
        if not PYPDF_AVAILABLE:
            return "", "pypdf is not installed. PDF ingestion is unavailable."
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text.strip())
            if not pages_text:
                return "", "No extractable text found in PDF (scanned or image-only PDF)."
            extracted = clean_text("\n\n".join(pages_text))
            return extracted, None
        except Exception as e:
            return "", f"Failed to parse PDF: {str(e)}"

    else:
        return "", f"Unsupported file extension '{ext}'. Only .txt and .pdf files are supported."


def segment_into_passages(
    text: str,
    doc_name: str,
    chunk_size: int = 150,
    overlap: int = 30
) -> List[Dict[str, Any]]:
    """
    Segment document text into sliding-window passages based on word count.
    
    Args:
        text: The document text to segment.
        doc_name: Identifier of the source document.
        chunk_size: Maximum word count per passage.
        overlap: Number of overlapping words between consecutive passages.
        
    Returns:
        List of passage dicts with metadata.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if not words:
        return []

    # Clamp parameters to valid ranges
    chunk_size = max(20, chunk_size)
    overlap = max(0, min(overlap, chunk_size - 10))
    step = chunk_size - overlap

    passages: List[Dict[str, Any]] = []
    passage_idx = 0

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        passages.append({
            "passage_id": f"{doc_name}_p{passage_idx}",
            "doc_name": doc_name,
            "passage_index": passage_idx,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_word_idx": i,
            "end_word_idx": i + len(chunk_words)
        })
        passage_idx += 1

        # If we have reached the end of the text, stop
        if i + chunk_size >= len(words):
            break

    return passages


class DocumentRepository:
    """
    In-memory document repository storing raw documents and indexed passages.
    """

    def __init__(self, default_chunk_size: int = 150, default_overlap: int = 30):
        self.documents: Dict[str, str] = {}
        self.chunk_size = default_chunk_size
        self.overlap = default_overlap
        self.passages: List[Dict[str, Any]] = []

    def set_chunk_parameters(self, chunk_size: int, overlap: int) -> None:
        """Update chunk size and overlap parameters and rebuild passages."""
        self.chunk_size = max(20, chunk_size)
        self.overlap = max(0, min(overlap, self.chunk_size - 10))
        self.rebuild_passages()

    def add_document(self, filename: str, content: str) -> Tuple[bool, str]:
        """
        Add or update a document in the repository.
        Returns (success, message).
        """
        cleaned = clean_text(content)
        if not cleaned:
            return False, f"Document '{filename}' is empty or contains only whitespace."

        self.documents[filename] = cleaned
        self.rebuild_passages()
        return True, f"Document '{filename}' successfully ingested ({len(cleaned.split())} words)."

    def remove_document(self, filename: str) -> bool:
        """Remove a document by filename."""
        if filename in self.documents:
            del self.documents[filename]
            self.rebuild_passages()
            return True
        return False

    def clear(self) -> None:
        """Clear all stored documents and passages."""
        self.documents.clear()
        self.passages.clear()

    def rebuild_passages(self) -> None:
        """Re-segment all stored documents into passages."""
        new_passages: List[Dict[str, Any]] = []
        for doc_name, content in self.documents.items():
            doc_passages = segment_into_passages(
                text=content,
                doc_name=doc_name,
                chunk_size=self.chunk_size,
                overlap=self.overlap
            )
            new_passages.extend(doc_passages)
        self.passages = new_passages

    def get_document_names(self) -> List[str]:
        return sorted(list(self.documents.keys()))

    def get_document_count(self) -> int:
        return len(self.documents)

    def get_passage_count(self) -> int:
        return len(self.passages)

    def load_sample_documents(self, sample_dir: str) -> int:
        """
        Load sample text files from a directory into the repository.
        Returns count of loaded documents.
        """
        if not os.path.exists(sample_dir):
            return 0

        count = 0
        for fname in sorted(os.listdir(sample_dir)):
            if fname.endswith(".txt") or fname.endswith(".pdf"):
                fpath = os.path.join(sample_dir, fname)
                try:
                    with open(fpath, "rb") as f:
                        data = f.read()
                    text, err = extract_text_from_bytes(data, fname)
                    if text and not err:
                        self.documents[fname] = text
                        count += 1
                except Exception:
                    pass

        self.rebuild_passages()
        return count
