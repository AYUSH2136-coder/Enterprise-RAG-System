# src/ingestion/text_chunker.py
# Responsibility: Split cleaned text into overlapping chunks using sliding window.

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(cleaned_text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """
    Splits a large text into smaller overlapping chunks.

    Why overlapping?
        If a sentence spans the boundary of two chunks, overlap ensures
        it appears fully in at least one chunk, preserving context.

    Why RecursiveCharacterTextSplitter?
        It tries to split at natural boundaries in this order:
        paragraph (\n\n) → line (\n) → sentence (". ") → word (" ") → character
        This keeps semantically related sentences together.

    Args:
        cleaned_text: Output from text_cleaner.clean_text()
        chunk_size:   Max characters per chunk (default 500)
        chunk_overlap: Characters shared between consecutive chunks (default 100)

    Returns:
        A list of text chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(cleaned_text)

    print(f"[text_chunker] Created {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks