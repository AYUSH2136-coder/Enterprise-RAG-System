# src/retrieval/retriever.py
# Responsibility: Given a user query, find and return the most relevant chunks.

import faiss
import numpy as np
from src.embedding.embedder import embed_query
from src.vectorstore.faiss_store import search_index


def retrieve(query: str, index: faiss.Index, chunks: list[str],
             top_k: int = 3) -> list[dict]:
    """
    Full retrieval flow for a single user query:
        1. Embed the query text into a vector
        2. Search FAISS index for top_k nearest vectors
        3. Map result indices back to original chunk text

    Args:
        query:  The user's natural language question.
        index:  Loaded FAISS index.
        chunks: List of original text chunks (same order as index).
        top_k:  Number of relevant chunks to retrieve.

    Returns:
        List of dicts, each containing:
            - rank       : 1-based position (1 = most relevant)
            - chunk_index: Position in the chunks list
            - distance   : L2 distance (lower = more similar)
            - text       : The actual chunk text
    """
    # Step 1: Convert query to vector
    query_vector = embed_query(query)

    # Step 2: Search the index
    distances, indices = search_index(index, query_vector, top_k)

    # Step 3: Build result list
    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        results.append({
            "rank": rank + 1,
            "chunk_index": int(idx),
            "distance": float(dist),
            "text": chunks[int(idx)]
        })

    print(f"[retriever] Retrieved top-{top_k} chunks for query: '{query[:60]}...'")
    return results