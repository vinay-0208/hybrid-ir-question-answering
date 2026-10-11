"""
Unit tests for document ingestion, text extraction, and passage segmentation.
"""

import pytest
from app.ingestion import (
    clean_text,
    extract_text_from_bytes,
    segment_into_passages,
    DocumentRepository
)


def test_clean_text():
    raw = "  Hello \t\t world!\r\n\r\n\r\nThis is a   test.  "
    cleaned = clean_text(raw)
    assert cleaned == "Hello world!\n\nThis is a test."
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_extract_text_from_bytes_txt():
    # Valid UTF-8
    data = "Natural Language Processing with Transformers.".encode("utf-8")
    text, err = extract_text_from_bytes(data, "sample.txt")
    assert err is None
    assert text == "Natural Language Processing with Transformers."

    # Empty bytes
    text, err = extract_text_from_bytes(b"", "empty.txt")
    assert err is not None
    assert text == ""

    # Unsupported extension
    text, err = extract_text_from_bytes(b"data", "file.docx")
    assert err is not None
    assert "Unsupported file extension" in err


def test_segment_into_passages():
    # 100 words text
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)

    # Chunk size 40, overlap 10 -> step = 30
    passages = segment_into_passages(text, "doc1.txt", chunk_size=40, overlap=10)
    assert len(passages) >= 3
    assert passages[0]["passage_id"] == "doc1.txt_p0"
    assert passages[0]["word_count"] == 40
    assert passages[0]["doc_name"] == "doc1.txt"

    # Edge cases
    assert segment_into_passages("", "empty.txt") == []
    assert segment_into_passages("Single sentence.", "doc.txt", chunk_size=100) != []


def test_document_repository():
    repo = DocumentRepository(default_chunk_size=50, default_overlap=10)
    assert repo.get_document_count() == 0
    assert repo.get_passage_count() == 0

    # Ingest document
    ok, msg = repo.add_document("doc1.txt", "This is the first document with useful content. " * 5)
    assert ok is True
    assert repo.get_document_count() == 1
    assert repo.get_passage_count() > 0

    # Ingest empty document
    ok_empty, msg_empty = repo.add_document("empty.txt", "   ")
    assert ok_empty is False

    # Remove document
    assert repo.remove_document("doc1.txt") is True
    assert repo.get_document_count() == 0
    assert repo.get_passage_count() == 0
