"""
product_knowledge.py
────────────────────
Agent 2: Product Knowledge Agent
RAG — HuggingFace embeddings + FAISS + Azure OpenAI GPT-4o.
"""

import logging
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.agents.vector_store import get_retriever

load_dotenv()
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Product Knowledge AI for a Smart Retail store.
Answer questions about products, pricing, return policies, restocking,
and store operations based on the provided knowledge base context.
Always base your answer on the context provided.
If context does not contain enough information, say so honestly.
Be concise, clear, and helpful."""


def _get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version="2024-02-01",
        temperature=0.2,
        max_tokens=600,
    )


def run(query: str, session_id: str = "") -> dict:
    log.info(f"[ProductKnowledgeAgent] query='{query[:80]}'")

    try:
        retriever = get_retriever(k=4)
        docs      = retriever.invoke(query)
        log.info(f"[ProductKnowledgeAgent] retrieved {len(docs)} chunks")
    except Exception as e:
        log.error(f"[ProductKnowledgeAgent] retrieval failed: {e}")
        return {"agent": "product_knowledge", "response": f"Retrieval error: {str(e)}", "status": "error", "sources": []}

    context     = "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    ])
    full_system = f"{SYSTEM_PROMPT}\n\nKNOWLEDGE BASE CONTEXT:\n\n{context}"

    try:
        llm      = _get_llm()
        messages = [SystemMessage(content=full_system), HumanMessage(content=query)]
        response = llm.invoke(messages)
        sources  = list({d.metadata.get("source", "") for d in docs})
        return {"agent": "product_knowledge", "response": response.content, "status": "success", "sources": sources}
    except Exception as e:
        log.error(f"[ProductKnowledgeAgent] LLM failed: {e}")
        return {"agent": "product_knowledge", "response": f"Error: {str(e)}", "status": "error", "sources": []}