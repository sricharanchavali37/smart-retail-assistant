"""
build_index.py
──────────────
Run this ONCE to build the FAISS vector index from knowledge documents.
The index is saved to backend/agents/faiss_index/ and reused on every
server start (no rebuild needed unless documents change).

Run from project root:
    python backend/agents/build_index.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
log = logging.getLogger(__name__)


def main():
    log.info("=" * 55)
    log.info("  BUILDING FAISS VECTOR INDEX")
    log.info("  Model: all-MiniLM-L6-v2 (HuggingFace)")
    log.info("=" * 55)

    from backend.agents.vector_store import build_index, get_retriever, DOCS_DIR, INDEX_DIR

    # List documents to be indexed
    docs = list(DOCS_DIR.glob("*.txt"))
    log.info(f"\n  Documents to index ({len(docs)} files):")
    for d in docs:
        log.info(f"    {d.name}")

    # Build
    log.info("\nBuilding index (downloading model on first run ~80MB)...")
    vectorstore = build_index()

    # Test retrieval
    log.info("\nTesting retrieval...")
    retriever = get_retriever(k=3)

    test_queries = [
        "return policy for electronics",
        "beauty product demand forecast",
        "anomaly detection threshold",
    ]

    for q in test_queries:
        results = retriever.invoke(q)
        log.info(f"\n  Query: '{q}'")
        for r in results:
            log.info(f"    [{r.metadata['source']}]  {r.page_content[:80]}...")

    log.info(f"\n  Index saved to: {INDEX_DIR}")
    log.info("\nDone. FAISS index is ready.")


if __name__ == "__main__":
    main()