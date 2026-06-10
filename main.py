# main.py
# Central entry point for the entire RAG pipeline.
# Run: python main.py

import os
import sys
import time

# ── Make sure src/ is importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.ingest_pipeline import run_ingestion
from src.pipeline.rag_pipeline import run_rag_query


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change these two values before running
# ──────────────────────────────────────────────────────────────────────────────

PDF_PATH    = input("📄 Enter the absolute path to your PDF document (e.g., data/raw/your_document.pdf): ").strip()
CONFIG_PATH = "config.yaml"
VECTOR_DB   = "data/vector_db/index.faiss"  # Used to detect if ingestion already done


# ──────────────────────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(title: str):
    print("\n" + "═" * 55)
    print(f"  {title}")
    print("═" * 55)


def is_already_ingested() -> bool:
    """Returns True if a vector DB already exists on disk."""
    return os.path.exists(VECTOR_DB)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — INGESTION
# ──────────────────────────────────────────────────────────────────────────────

def run_stage_ingestion():
    print_banner("STAGE 1 — Document Ingestion")

    if is_already_ingested():
        print("⚡ Vector database already exists. Skipping ingestion.")
        print(f"   (Delete '{VECTOR_DB}' to force re-ingestion)")
        return

    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found at: {PDF_PATH}")
        print("   Please place your PDF inside data/raw/ and update PDF_PATH in main.py")
        sys.exit(1)

    start = time.time()
    run_ingestion(pdf_path=PDF_PATH, config_path=CONFIG_PATH)
    elapsed = round(time.time() - start, 2)

    print(f"\n✅ Ingestion complete in {elapsed}s")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — INTERACTIVE Q&A LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_stage_query():
    print_banner("STAGE 2 — Interactive Q&A")
    print("Type your question and press Enter.")
    print("Commands:  'quit' → exit  |  'reingest' → re-run ingestion\n")

    while True:
        try:
            query = input("❓ Your Question:(Type 'quit' to exit) ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Exiting. Goodbye!")
            sys.exit(0)

        # ── Handle special commands ────────────────────────────────────────
        if not query:
            continue

        if query.lower() == "quit":
            print("👋 Exiting. Goodbye!")
            sys.exit(0)

        if query.lower() == "reingest":
            print("🔄 Forcing re-ingestion...")
            # Remove old index so run_stage_ingestion() doesn't skip it
            if os.path.exists(VECTOR_DB):
                os.remove(VECTOR_DB)
                chunks_path = VECTOR_DB.replace("index.faiss", "chunks.json")
                if os.path.exists(chunks_path):
                    os.remove(chunks_path)
            run_stage_ingestion()
            continue

        # ── Normal query ───────────────────────────────────────────────────
        print("\n⏳ Searching documents and generating answer...\n")
        start = time.time()

        try:
            answer = run_rag_query(query=query, config_path=CONFIG_PATH)
            elapsed = round(time.time() - start, 2)

            print("─" * 55)
            print("🤖 Answer:")
            print(answer)
            print(f"\n⏱  Generated in {elapsed}s")
            print("─" * 55)

        except Exception as e:
            print(f"❌ Error during query: {e}")
            print("   Make sure ingestion ran successfully first.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print_banner("🧠 Enterprise RAG System")
    print(f"  Document : {PDF_PATH}")
    print(f"  Config   : {CONFIG_PATH}")

    run_stage_ingestion()   # Step 1: Ingest PDF (skips if already done)
    run_stage_query()       # Step 2: Start Q&A loop


if __name__ == "__main__":
    main()