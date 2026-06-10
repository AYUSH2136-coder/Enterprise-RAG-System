# src/vectorstore/faiss_store.py
# Responsibility: Build, save, load, and search the FAISS vector index.

import faiss
import numpy as np
import json
import os

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Creates a FAISS flat index and adds all chunk embeddings to it.

    IndexFlatL2: 
        - "Flat" = stores all vectors; does exact (not approximate) search
        - "L2"   = uses Euclidean distance to measure similarity
        - Best choice for datasets up to ~100k chunks

    Args:
        embeddings: numpy array of shape (num_chunks, embedding_dim)

    Returns:
        A FAISS index containing all vectors.
    """
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"[faiss_store] Built index with {index.ntotal} vectors (dim={dimension})")
    return index


def save_index(index: faiss.Index, chunks: list[str], save_dir: str):
    """
    Persists the FAISS index and the original text chunks to disk.

    Why save chunks separately?
        FAISS stores only vectors. To return actual text answers,
        we need the original chunk strings mapped by index position.

    Args:
        index:    The FAISS index object.
        chunks:   List of text strings (same order as embeddings).
        save_dir: Directory path where files will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    index_path = os.path.join(save_dir, "index.faiss")
    chunks_path = os.path.join(save_dir, "chunks.json")

    faiss.write_index(index, index_path)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[faiss_store] Saved index → {index_path}")
    print(f"[faiss_store] Saved chunks → {chunks_path}")


def load_index(save_dir: str) -> tuple[faiss.Index, list[str]]:
    """
    Loads a previously saved FAISS index and its corresponding chunks.

    Args:
        save_dir: Directory where index.faiss and chunks.json are stored.

    Returns:
        Tuple of (faiss_index, list_of_chunk_strings)
    """
    index_path = os.path.join(save_dir, "index.faiss")
    chunks_path = os.path.join(save_dir, "chunks.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No FAISS index found at: {index_path}")

    index = faiss.read_index(index_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[faiss_store] Loaded index with {index.ntotal} vectors")
    return index, chunks


def search_index(index: faiss.Index, query_vector: np.ndarray,
                 top_k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Searches the FAISS index for the top_k most similar vectors.

    Args:
        index:        The FAISS index to search.
        query_vector: numpy array of shape (1, embedding_dim)
        top_k:        Number of results to return.

    Returns:
        Tuple of (distances, indices) — both numpy arrays of shape (1, top_k)
        Lower distance = more similar.
    """
    distances, indices = index.search(query_vector, top_k)
    return distances, indices