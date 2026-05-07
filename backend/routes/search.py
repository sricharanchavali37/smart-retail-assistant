"""
search.py
─────────
POST /search
Document search using Azure AI Search (vector + keyword hybrid search).
Searches product catalogs, FAQs, and retail knowledge base.

Request body:
  {
    "query": "what are the top beauty products?",
    "top_k": 3
  }

Response: top-k matching documents with content + metadata
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()
log = logging.getLogger(__name__)
router = APIRouter()

# ── Azure AI Search credentials from .env ─────────────────────────────────────
AZURE_SEARCH_ENDPOINT   = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_API_KEY    = os.getenv("AZURE_SEARCH_API_KEY", "")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "retail-knowledge")

try:
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    log.warning("azure-search-documents not installed. Search will use fallback.")


# ── Request / Response schemas ────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str          = Field(..., description="Natural language search query", min_length=1)
    top_k: int          = Field(3,  description="Number of results to return",   ge=1, le=10)
    category_filter: Optional[str] = Field(None, description="Filter by product category")


# ── Azure AI Search client ────────────────────────────────────────────────────
def get_search_client():
    if not SEARCH_AVAILABLE:
        return None
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
        return None
    try:
        return SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
        )
    except Exception as e:
        log.error(f"SearchClient init failed: {e}")
        return None


# ── Fallback knowledge base (used when Azure AI Search is not configured) ────
FALLBACK_KB = [
    {
        "id": "001",
        "title": "Beauty Category Overview",
        "content": (
            "The Beauty category includes skincare, cosmetics, and personal care products. "
            "Top performers include moisturisers, serums, and makeup kits. "
            "Average price point: Rs 25 - Rs 500 per unit. "
            "High demand months: February (Valentine's), October-December (festive season)."
        ),
        "category": "Beauty",
        "score": 0.95,
    },
    {
        "id": "002",
        "title": "Electronics Category Overview",
        "content": (
            "Electronics includes gadgets, accessories, and consumer electronics. "
            "High-value items: Rs 300 - Rs 500 per unit. "
            "Most volatile category — prone to demand spikes during sales events. "
            "Top anomaly category with 30 flagged days in 2023 analysis."
        ),
        "category": "Electronics",
        "score": 0.92,
    },
    {
        "id": "003",
        "title": "Clothing Category Overview",
        "content": (
            "Clothing includes apparel, accessories, and fashion items. "
            "Price range: Rs 25 - Rs 500 per unit. "
            "Steady demand throughout the year with moderate seasonal variation. "
            "Weekend sales tend to be higher than weekday sales."
        ),
        "category": "Clothing",
        "score": 0.90,
    },
    {
        "id": "004",
        "title": "Demand Forecasting Methodology",
        "content": (
            "The Smart Retail Assistant uses XGBoost with 29 engineered features "
            "for demand forecasting. Validation MAPE: 8.87%. "
            "Features include lag values (1, 7, 14 days), rolling averages, "
            "day-of-week effects, and customer demographic signals."
        ),
        "category": "All",
        "score": 0.88,
    },
    {
        "id": "005",
        "title": "Anomaly Detection Policy",
        "content": (
            "Anomalies are detected using Isolation Forest with 5% contamination rate. "
            "53 anomalous days were identified in 2023 (5.02% of records). "
            "Electronics had the most anomalies (30), driven by high price variance. "
            "Anomaly alerts trigger automatic review in the retail dashboard."
        ),
        "category": "All",
        "score": 0.85,
    },
    {
        "id": "006",
        "title": "Return and Refund Policy",
        "content": (
            "Products can be returned within 30 days of purchase with original receipt. "
            "Electronics must be unused and in original packaging. "
            "Beauty products are non-returnable once opened. "
            "Clothing returns accepted if tags are intact."
        ),
        "category": "All",
        "score": 0.80,
    },
]


def _fallback_search(query: str, top_k: int, category_filter: Optional[str]) -> list:
    """Simple keyword match fallback when Azure AI Search is not configured."""
    query_lower = query.lower()
    results = []
    for doc in FALLBACK_KB:
        if category_filter and doc["category"] not in [category_filter, "All"]:
            continue
        score = 0
        for word in query_lower.split():
            if word in doc["content"].lower() or word in doc["title"].lower():
                score += 1
        results.append({**doc, "relevance_score": score + doc["score"]})
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_k]


@router.post(
    "/",
    summary="Search product knowledge base",
    description=(
        "Search retail knowledge documents using Azure AI Search. "
        "Falls back to local knowledge base if Azure not configured."
    ),
)
async def search(request: SearchRequest):
    log.info(f"POST /search  query='{request.query}'  top_k={request.top_k}")

    client = get_search_client()

    # ── Azure AI Search ──────────────────────────────────────────
    if client is not None:
        try:
            filter_expr = None
            if request.category_filter:
                filter_expr = f"category eq '{request.category_filter}'"

            results = client.search(
                search_text=request.query,
                top=request.top_k,
                filter=filter_expr,
                include_total_count=True,
            )
            docs = []
            for r in results:
                docs.append({
                    "id":              r.get("id", ""),
                    "title":           r.get("title", ""),
                    "content":         r.get("content", ""),
                    "category":        r.get("category", ""),
                    "relevance_score": r.get("@search.score", 0),
                    "source":          "azure_ai_search",
                })
            log.info(f"  Azure AI Search returned {len(docs)} results")
            return JSONResponse(content={
                "query":   request.query,
                "results": docs,
                "count":   len(docs),
                "source":  "Azure AI Search",
            })
        except Exception as e:
            log.error(f"Azure AI Search failed: {e} — falling back to local KB")

    # ── Fallback ─────────────────────────────────────────────────
    log.info("  Using local knowledge base (Azure AI Search not configured)")
    docs = _fallback_search(request.query, request.top_k, request.category_filter)

    return JSONResponse(content={
        "query":   request.query,
        "results": docs,
        "count":   len(docs),
        "source":  "local_knowledge_base",
        "note":    "Configure AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY in .env for full vector search.",
    })