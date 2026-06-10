# src/pipeline/rag_pipeline.py
# Responsibility: Orchestrate the full query-to-answer RAG flow.

import yaml
from src.vectorstore.faiss_store import load_index
from src.retrieval.retriever import retrieve
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import generate_answer


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_rag_query(query: str, config_path: str = "config.yaml") -> str:
    """
    Full RAG query pipeline:
        User query -> embed -> retrieve chunks -> build prompt -> LLM -> answer

    Args:
        query: The user's natural language question.
        config_path: Path to config.yaml.

    Returns:
        The LLM-generated answer grounded in retrieved document chunks.
    """
    config = load_config(config_path)

    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print("="*50)

    print("\nSTAGE 1: Loading Vector Database")
    index, chunks = load_index(config['paths']['vector_db'])

    print("\nSTAGE 2: Retrieving Relevant Chunks")
    retrieved = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        top_k=config['retrieval']['top_k']
    )

    print("\nSTAGE 3: Building RAG Prompt")
    prompt = build_prompt(query, retrieved)

    print("\nSTAGE 4: Generating Answer via LLM")
    answer = generate_answer(
        prompt=prompt,
        model=config['llm']['model'],
        max_tokens=config['llm']['max_tokens']
    )

    return answer