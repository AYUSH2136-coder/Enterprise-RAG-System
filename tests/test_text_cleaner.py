# tests/test_text_cleaner.py

from src.ingestion.text_cleaner import clean_text

def test_removes_page_markers():
    raw = "--- PAGE 1 ---\nSome content here."
    result = clean_text(raw)
    assert "--- PAGE 1 ---" not in result
    assert "Some content here." in result

def test_removes_standalone_numbers():
    raw = "Introduction\n\n42\n\nThis is the content."
    result = clean_text(raw)
    assert "\n42\n" not in result

def test_collapses_extra_newlines():
    raw = "Line one\n\n\n\n\nLine two"
    result = clean_text(raw)
    assert "\n\n\n" not in result

def test_returns_stripped_string():
    raw = "   \n\nHello World\n\n   "
    result = clean_text(raw)
    assert result == result.strip()