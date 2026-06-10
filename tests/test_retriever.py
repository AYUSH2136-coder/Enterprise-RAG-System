# tests/test_retriever.py

import numpy as np
from src.embedding.embedder import embed_chunks
from src.vectorstore.faiss_store import build_index
from src.retrieval.retriever import retrieve

CHUNKS = [
    "The annual leave policy grants employees 20 days per year.",
    "Employees must apply for leave two weeks in advance via HR portal.",
    "Sick leave is separate and allows up to 10 days per year.",
    "The company cafeteria is open from 9am to 6pm on weekdays.",
    "Quarterly performance reviews happen in March, June, September, December."
]

def setup_test_index():
    embeddings = embed_chunks(CHUNKS)
    index = build_index(embeddings)
    return index

def test_retrieve_returns_list():
    index = setup_test_index()
    results = retrieve("How many leave days do I get?", index, CHUNKS, top_k=3)
    assert isinstance(results, list)
    assert len(results) == 3

def test_retrieve_top_result_is_relevant():
    index = setup_test_index()
    results = retrieve("What is the annual leave entitlement?", index, CHUNKS, top_k=3)
    # The top result should be about leave, not cafeteria or reviews
    top_text = results[0]['text'].lower()
    assert "leave" in top_text

def test_retrieve_result_structure():
    index = setup_test_index()
    results = retrieve("leave policy", index, CHUNKS, top_k=2)
    for r in results:
        assert "rank" in r
        assert "chunk_index" in r
        assert "distance" in r
        assert "text" in r