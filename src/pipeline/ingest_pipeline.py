# src/pipeline/ingest_pipeline.py
# Responsibility: Orchestrate the full document ingestion flow.

import yaml
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_cleaner import clean_text
from src.ingestion.text_chunker import chunk_text
from src.embedding.embedder import embed_chunks
from src.vectorstore.faiss_store import build_index, save_index


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_ingestion(pdf_path: str, config_path: str = "config.yaml"):
    """
    Full ingestion pipeline:
        PDF -> clean text -> chunks -> embeddings -> FAISS index -> saved to disk

    Args:
        pdf_path:    Path to the source PDF document.
        config_path: Path to config.yaml for settings.
    """
    config = load_config(config_path)

    print("\n" + "="*50)
    print("STAGE 1: Loading PDF")
    print("="*50)
    raw_text = load_pdf(pdf_path)

    print("\n" + "="*50)
    print("STAGE 2: Cleaning Text")
    print("="*50)
    cleaned_text = clean_text(raw_text)

    # Save cleaned text for inspection/debugging
    import os
    os.makedirs(config['paths']['processed_data'], exist_ok=True)
    processed_path = os.path.join(
        config['paths']['processed_data'], "cleaned_text.txt"
    )
    with open(processed_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"[ingest_pipeline] Cleaned text saved to {processed_path}")

    print("\n" + "="*50)
    print("STAGE 3: Chunking Text")
    print("="*50)
    chunks = chunk_text(
        cleaned_text,
        chunk_size=config['chunking']['chunk_size'],
        chunk_overlap=config['chunking']['chunk_overlap']
    )

    print("\n" + "="*50)
    print("STAGE 4: Generating Embeddings")
    print("="*50)
    embeddings = embed_chunks(
        chunks,
        model_name=config['embedding']['model_name'],
        batch_size=config['embedding']['batch_size']
    )

    print("\n" + "="*50)
    print("STAGE 5: Building & Saving Vector Index")
    print("="*50)
    index = build_index(embeddings)
    save_index(index, chunks, config['paths']['vector_db'])

    print("\n✅ Ingestion complete!")
    print(f"   Chunks stored: {len(chunks)}")
    print(f"   Vector DB saved to: {config['paths']['vector_db']}")