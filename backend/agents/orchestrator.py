"""
orchestrator.py
───────────────
LangChain multi-agent orchestrator.
Routes user queries to the correct agent based on intent detection.
Maintains per-session conversation memory.

Agents:
  retail_analyst     → sales data, trends, anomalies
  product_knowledge  → RAG over product docs and policies
  forecast_insight   → XGBoost demand forecasts + recommendations
"""

import logging
from typing import Optional

from backend.agents import retail_analyst, product_knowledge, forecast_insight

log = logging.getLogger(__name__)

# ── In-memory session history ─────────────────────────────────────────────────
# session_id -> list of {"role": "user"|"agent", "content": str}
_session_memory: dict = {}

MAX_HISTORY = 6   # keep last 6 turns per session


# ── Intent routing ────────────────────────────────────────────────────────────
FORECAST_KEYWORDS = [
    "forecast", "predict", "next week", "next month", "demand",
    "will sell", "revenue next", "stock up", "restock", "how much will",
    "14 day", "7 day", "days ahead", "upcoming",
]

ANALYST_KEYWORDS = [
    "anomaly", "anomalies", "spike", "unusual", "compare", "vs",
    "best selling", "top category", "performance", "trend",
    "which category", "how did", "last month", "2023", "revenue was",
    "highest", "lowest", "sales data",
]

KNOWLEDGE_KEYWORDS = [
    "return policy", "refund", "exchange", "warranty",
    "what is", "tell me about", "explain", "how does",
    "restocking", "lead time", "price", "product info",
    "policy", "customer", "loyalty",
]


def _detect_intent(message: str) -> str:
    """Return agent name based on keyword matching."""
    msg = message.lower()

    forecast_score  = sum(1 for kw in FORECAST_KEYWORDS  if kw in msg)
    analyst_score   = sum(1 for kw in ANALYST_KEYWORDS   if kw in msg)
    knowledge_score = sum(1 for kw in KNOWLEDGE_KEYWORDS if kw in msg)

    scores = {
        "forecast_insight":  forecast_score,
        "retail_analyst":    analyst_score,
        "product_knowledge": knowledge_score,
    }

    best = max(scores, key=scores.get)

    # Default to retail_analyst if all scores are 0
    if scores[best] == 0:
        best = "retail_analyst"

    log.info(f"  Intent scores: {scores} → routed to: {best}")
    return best


def _get_history(session_id: str) -> list:
    return _session_memory.get(session_id, [])


def _add_to_history(session_id: str, role: str, content: str):
    if session_id not in _session_memory:
        _session_memory[session_id] = []
    _session_memory[session_id].append({"role": role, "content": content})
    # Keep only last MAX_HISTORY turns
    if len(_session_memory[session_id]) > MAX_HISTORY * 2:
        _session_memory[session_id] = _session_memory[session_id][-MAX_HISTORY * 2:]


def run(message: str, session_id: Optional[str] = None) -> dict:
    """
    Main orchestrator entry point.
    Routes message to correct agent, manages memory, returns full response.
    """
    session_id = session_id or "default"
    log.info(f"[Orchestrator] session={session_id}  message='{message[:80]}'")

    # ── Detect intent ────────────────────────────────────────────
    agent_name = _detect_intent(message)

    # ── Save user message to memory ──────────────────────────────
    _add_to_history(session_id, "user", message)

    # ── Route to correct agent ───────────────────────────────────
    if agent_name == "forecast_insight":
        result = forecast_insight.run(message, session_id)
    elif agent_name == "product_knowledge":
        result = product_knowledge.run(message, session_id)
    else:
        result = retail_analyst.run(message, session_id)

    # ── Save agent response to memory ────────────────────────────
    _add_to_history(session_id, "agent", result.get("response", ""))

    # ── Build final response ─────────────────────────────────────
    history = _get_history(session_id)

    return {
        "session_id":       session_id,
        "message":          message,
        "agent":            agent_name,
        "response":         result.get("response", ""),
        "status":           result.get("status", "unknown"),
        "sources":          result.get("sources", []),
        "conversation_turns": len(history) // 2,
        "phase":            "3 of 6 — LangChain + Azure OpenAI active",
    }


def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in _session_memory:
        del _session_memory[session_id]
        log.info(f"[Orchestrator] Cleared session: {session_id}")