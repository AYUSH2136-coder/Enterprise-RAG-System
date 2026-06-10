# tests/test_embedder.py

import numpy as np
from src.embedding.embedder import embed_chunks, embed_query

SAMPLE_CHUNKS = ["The company leave policy allows 20 days per year.",
                 "Employees must submit requests two weeks in advance."]

def test_embed_chunks_returns_numpy_array():
    result = embed_chunks(SAMPLE_CHUNKS)
    assert isinstance(result, np.ndarray)

def test_embed_chunks_shape():
    result = embed_chunks(SAMPLE_CHUNKS)
    assert result.shape[0] == len(SAMPLE_CHUNKS)
    assert result.shape[1] == 384  # MiniLM-L6-v2 dimension

def test_embed_query_shape():
    result = embed_query("What is the leave policy?")
    assert result.shape == (1, 384)

def test_similar_sentences_closer_than_unrelated():
    e = embed_chunks([
        "Leave policy allows 20 days.",
        "Leave policy gives 20 annual days off.",
        "The quarterly revenue exceeded projections."
    ])
    # Distance between similar sentences should be less than unrelated ones
    dist_similar = np.linalg.norm(e[0] - e[1])
    dist_unrelated = np.linalg.norm(e[0] - e[2])
    assert dist_similar < dist_unrelated