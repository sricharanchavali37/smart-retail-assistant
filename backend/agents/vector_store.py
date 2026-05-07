"""
vector_store.py
───────────────
Builds and persists a FAISS vector store from local knowledge documents.
Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2) for embeddings.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

log = logging.getLogger(__name__)

AGENTS_DIR      = Path(__file__).resolve().parent
DOCS_DIR        = AGENTS_DIR / "documents"
INDEX_DIR       = AGENTS_DIR / "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _load_documents() -> list:
    documents = []
    for txt_file in sorted(DOCS_DIR.glob("*.txt")):
        text       = txt_file.read_text(encoding="utf-8")
        chunk_size = 500
        overlap    = 50
        start      = 0
        chunks     = []
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source":   txt_file.name,
                    "chunk_id": i,
                    "category": txt_file.stem.replace("_knowledge", "").replace("_", " ").title(),
                }
            ))
        log.info(f"  Loaded {len(chunks)} chunks from {txt_file.name}")
    log.info(f"  Total chunks: {len(documents)}")
    return documents


def _get_embeddings():
    log.info(f"Loading HuggingFace embeddings: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index() -> FAISS:
    log.info("Building FAISS index...")
    documents   = _load_documents()
    embeddings  = _get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    log.info(f"FAISS index saved to {INDEX_DIR}")
    return vectorstore


def load_index() -> FAISS:
    log.info(f"Loading FAISS index from {INDEX_DIR}")
    embeddings = _get_embeddings()
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_vectorstore() -> FAISS:
    if (INDEX_DIR / "index.faiss").exists():
        try:
            return load_index()
        except Exception as e:
            log.warning(f"Failed to load index: {e} — rebuilding...")
    return build_index()


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_index()
    retriever = get_retriever()
    results   = retriever.invoke("return policy for electronics")
    for r in results:
        log.info(f"  [{r.metadata['source']}]  {r.page_content[:100]}...")