# src/ingestion/text_cleaner.py
# Responsibility: Remove noise from raw PDF text using regex and NLP.

import re
import nltk

# Download required NLTK data (runs only once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(raw_text: str) -> str:

    # Step 1: Remove page markers we inserted during loading
    text = re.sub(r'--- PAGE \d+ ---', '', raw_text)

    # Step 2: Remove lines that contain only a number (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Step 3: Collapse 3+ consecutive newlines into just 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Step 4: Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text)

    # Step 5: Remove non-ASCII characters (e.g., \x0c form feeds, weird symbols)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Step 6: Final strip
    text = text.strip()

    print(f"[text_cleaner] Cleaned text length: {len(text)} characters")
    return text