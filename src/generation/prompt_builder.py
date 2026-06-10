# src/generation/prompt_builder.py
# Responsibility: Assemble the final prompt that gets sent to the LLM.

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Constructs a RAG prompt by injecting retrieved context into a template.

    The critical instruction "Answer ONLY from the context below" is what
    prevents the LLM from hallucinating outside information.

    Args:
        query:            The user's original question.
        retrieved_chunks: Output from retriever.retrieve() — list of dicts
                          with 'rank' and 'text' keys.

    Returns:
        A formatted string ready to be sent to the LLM.
    """
    # Build the context block from all retrieved chunks
    context_sections = []
    for chunk in retrieved_chunks:
        section = f"[Relevant Section {chunk['rank']}]\n{chunk['text']}"
        context_sections.append(section)

    context_block = "\n\n".join(context_sections)

    prompt = f"""You are a precise and trustworthy enterprise assistant.

Your task is to answer the user's question using ONLY the information 
provided in the context sections below.

Rules:
- Base your answer exclusively on the provided context.
- If the answer is not present in the context, respond with:
  "I could not find this information in the provided documents."
- Do NOT use any outside knowledge or make assumptions.
- Keep the answer clear, professional, and concise.

=== CONTEXT FROM INTERNAL DOCUMENTS ===

{context_block}

=== USER QUESTION ===
{query}

=== YOUR ANSWER ==="""

    return prompt