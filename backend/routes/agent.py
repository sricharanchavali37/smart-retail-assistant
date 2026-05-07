"""
agent.py  (Phase 3 — updated)
──────────────────────────────
POST /agent
Fully wired to LangChain multi-agent orchestrator.
Routes to retail_analyst, product_knowledge, or forecast_insight
based on intent detection.

Request body:
  {
    "message":    "Which category will have highest demand next week?",
    "session_id": "user-123"
  }
"""

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.agents.orchestrator import run as orchestrator_run
from backend.agents.orchestrator import clear_session

log = logging.getLogger(__name__)
router = APIRouter()


class AgentRequest(BaseModel):
    message:    str           = Field(..., min_length=1, description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for memory")


class ClearSessionRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to clear")


@router.post(
    "/",
    summary="Chat with AI agents",
    description=(
        "Routes your message to the correct agent: "
        "Retail Analyst, Product Knowledge (RAG), or Forecast Insight. "
        "Powered by LangChain + Azure OpenAI GPT-4o."
    ),
)
async def agent(request: AgentRequest):
    log.info(f"POST /agent  session={request.session_id}  msg='{request.message[:60]}'")

    result = orchestrator_run(
        message=request.message,
        session_id=request.session_id,
    )

    return JSONResponse(content=result)


@router.post(
    "/clear",
    summary="Clear conversation memory",
    description="Clears the conversation history for a given session ID.",
)
async def clear(request: ClearSessionRequest):
    clear_session(request.session_id)
    return JSONResponse(content={
        "message":    f"Session {request.session_id} cleared.",
        "session_id": request.session_id,
    })


@router.get(
    "/agents",
    summary="List available agents",
    description="Returns the 3 available agents and what they do.",
)
async def list_agents():
    return JSONResponse(content={
        "agents": [
            {
                "name":        "retail_analyst",
                "description": "Analyses sales data, trends, anomalies from retail_features.csv",
                "example":     "Which category had the highest revenue last month?",
            },
            {
                "name":        "product_knowledge",
                "description": "RAG over product catalog, policies, restocking guides (FAISS + HuggingFace)",
                "example":     "What is the return policy for Electronics?",
            },
            {
                "name":        "forecast_insight",
                "description": "Interprets XGBoost demand forecasts and gives stock recommendations",
                "example":     "Should I stock up on Clothing next week?",
            },
        ]
    })