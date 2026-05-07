"""
main.py
───────
FastAPI application entry point for the Smart Retail Assistant.

APIs:
  POST /ingest   - upload sales CSV, store to Azure SQL + Blob
  GET  /predict  - XGBoost demand forecast per category
  POST /search   - Azure AI Search over product documents
  POST /agent    - LangChain multi-agent orchestrator (Phase 3)

Run locally:
  uvicorn backend.main:app --reload --port 8000
Or from backend/:
  uvicorn main:app --reload --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Path fix so imports work from any working directory ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.routes.ingest  import router as ingest_router
from backend.routes.predict import router as predict_router
from backend.routes.search  import router as search_router
from backend.routes.agent   import router as agent_router

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("retail_api.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 55)
    log.info("  Smart Retail Assistant API starting up...")
    log.info("  Docs: http://localhost:9000/docs")
    log.info("=" * 55)
    yield
    log.info("Smart Retail Assistant API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Retail Assistant API",
    description=(
        "Multi-agent retail platform: demand forecasting, "
        "customer Q&A, and anomaly detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict in production via env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest_router,  prefix="/ingest",  tags=["Data Ingestion"])
app.include_router(predict_router, prefix="/predict", tags=["Demand Forecasting"])
app.include_router(search_router,  prefix="/search",  tags=["Document Search"])
app.include_router(agent_router,   prefix="/agent",   tags=["AI Agents"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "Smart Retail Assistant API"}


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Smart Retail Assistant API",
        "docs":    "http://localhost:9000/docs",
        "endpoints": {
            "POST /ingest":  "Upload sales CSV data",
            "GET  /predict": "Demand forecast by category",
            "POST /search":  "Search product documents",
            "POST /agent":   "Chat with AI agents",
        },
    }


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)