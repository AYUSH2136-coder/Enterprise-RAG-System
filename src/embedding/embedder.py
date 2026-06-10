# src/embedding/embedder.py
# Responsibility: Convert text chunks into numerical vectors (embeddings).

import numpy as np
from sentence_transformers import SentenceTransformer

# Load model once at module level so it isn't reloaded on every function call
_model = None

def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads and caches the embedding model.
    Downloads the model on first run (~90MB), then uses local cache.
    """
    global _model
    if _model is None:
        print(f"[embedder] Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def embed_chunks(chunks: list[str], model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32) -> np.ndarray:
    """
    Converts a list of text chunks into a 2D numpy array of embeddings.

    Each chunk becomes a vector of 384 floats (for MiniLM-L6-v2).
    Similar text → similar vectors → close together in vector space.

    Args:
        chunks:     List of text strings to embed.
        model_name: HuggingFace model identifier.
        batch_size: Number of chunks processed in one forward pass.

    Returns:
        numpy array of shape (num_chunks, embedding_dim) as float32.
    """
    model = get_model(model_name)

    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = np.array(embeddings, dtype='float32')
    print(f"[embedder] Embedding matrix shape: {embeddings.shape}")
    return embeddings


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embeds a single query string into a vector.
    Used at query time to convert user question → searchable vector.

    Returns:
        numpy array of shape (1, embedding_dim) as float32.
    """
    model = get_model(model_name)
    vector = model.encode([query], convert_to_numpy=True)
    return np.array(vector, dtype='float32')