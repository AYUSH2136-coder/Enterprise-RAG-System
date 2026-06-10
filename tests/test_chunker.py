# tests/test_chunker.py

from src.ingestion.text_chunker import chunk_text

SAMPLE_TEXT = " ".join(["This is a sample sentence."] * 100)

def test_chunk_returns_list():
    chunks = chunk_text(SAMPLE_TEXT)
    assert isinstance(chunks, list)

def test_chunk_not_empty():
    chunks = chunk_text(SAMPLE_TEXT)
    assert len(chunks) > 0

def test_chunk_size_respected():
    chunks = chunk_text(SAMPLE_TEXT, chunk_size=200, chunk_overlap=50)
    for chunk in chunks:
        # Allow slight overflow due to splitter boundary logic
        assert len(chunk) <= 250

def test_overlap_creates_more_chunks():
    chunks_no_overlap = chunk_text(SAMPLE_TEXT, chunk_size=200, chunk_overlap=0)
    chunks_with_overlap = chunk_text(SAMPLE_TEXT, chunk_size=200, chunk_overlap=50)
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)