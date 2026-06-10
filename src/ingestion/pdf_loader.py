# src/ingestion/pdf_loader.py
# Responsibility: Open a PDF file and extract raw text from every page.

import fitz  # PyMuPDF
import os

def load_pdf(pdf_path: str) -> str:
    """
    Opens a PDF and extracts all text page by page.
    
    Args:
        pdf_path: Absolute or relative path to the PDF file.
    
    Returns:
        A single string containing all extracted text.
    
    Raises:
        FileNotFoundError: If the PDF doesn't exist at the given path.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")  # "text" = plain text mode
        full_text += f"\n\n--- PAGE {page_num + 1} ---\n\n{page_text}"

    doc.close()

    print(f"[pdf_loader] Extracted {len(full_text)} characters from {pdf_path}")
    return full_text