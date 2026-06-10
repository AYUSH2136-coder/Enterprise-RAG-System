# tests/test_pdf_loader.py

import pytest
from src.ingestion.pdf_loader import load_pdf

def test_load_pdf_file_not_found():
    """Should raise FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_pdf("data/raw/nonexistent.pdf")

def test_load_pdf_returns_string(tmp_path):
    """Should return a non-empty string for a valid PDF."""
    # Use a real small PDF in your test fixtures for a real test
    # This is a placeholder structure
    pass