# tests/test_rag_pipeline.py

from src.generation.prompt_builder import build_prompt

MOCK_CHUNKS = [
    {"rank": 1, "text": "Employees get 20 days of annual leave.", "distance": 0.12},
    {"rank": 2, "text": "Leave must be applied two weeks in advance.", "distance": 0.25},
]

def test_prompt_contains_query():
    query = "How many leave days do I get?"
    prompt = build_prompt(query, MOCK_CHUNKS)
    assert query in prompt

def test_prompt_contains_chunk_text():
    prompt = build_prompt("test query", MOCK_CHUNKS)
    assert "20 days of annual leave" in prompt
    assert "two weeks in advance" in prompt

def test_prompt_contains_restriction_instruction():
    prompt = build_prompt("test query", MOCK_CHUNKS)
    assert "ONLY" in prompt  # The hallucination-prevention instruction