# scripts/ingest_document.py
# Entry point: To ingest a PDF into the vector database.
# Usage: python scripts/ingest_document.py --pdf data/raw/manual.pdf

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.ingest_pipeline import run_ingestion

def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG vector database.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to ingest")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    run_ingestion(pdf_path=args.pdf, config_path=args.config)

if __name__ == "__main__":
    main()