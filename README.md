# Smart Retail Assistant
### Multi-Agent AI Platform — Demand Forecasting + GenAI Agents + Azure

[![CI/CD](https://github.com/sricharanchavali37/smart-retail-assistant/actions/workflows/deploy.yml/badge.svg)](https://github.com/sricharanchavali37/smart-retail-assistant/actions)

## Live API
- **URL:** https://smartretailapi.lemonmushroom-09a9d8f9.koreacentral.azurecontainerapps.io
- **Docs:** https://smartretailapi.lemonmushroom-09a9d8f9.koreacentral.azurecontainerapps.io/docs

---

## Project Overview

Smart Retail Assistant is a production-grade multi-agent AI platform for retail analytics. It combines machine learning demand forecasting, anomaly detection, and LangChain-powered AI agents backed by Azure OpenAI GPT-4o.

**Dataset:** 1,000 retail transactions (2023) across Beauty, Clothing, and Electronics categories.

---

## Architecture



retail_sales.csv
│
▼
Phase 1 — ML Models
XGBoost Demand Forecasting   MAPE 8.87%
Isolation Forest Anomaly     53 anomalies detected
│
▼
Phase 2 — FastAPI Backend (port 9000)
POST /ingest   — CSV upload
GET  /predict  — 7/14 day forecast
POST /search   — knowledge base search
POST /agent    — AI agent routing
│
▼
Phase 3 — LangChain Multi-Agent System
Retail Analyst Agent     — sales data Q&A
Product Knowledge Agent  — RAG over docs (FAISS + HuggingFace)
Forecast Insight Agent   — XGBoost + GPT-4o recommendations
│
▼
Phase 4 — Azure Data Pipeline
Azure Blob Storage      — bronze container (5 files)
Azure Data Factory      — Medallion pipeline defined
Power BI Dashboard      — 4 pages
│
▼
Phase 5 — Deployment
Docker Container        — sricharanchavali37/smart-retail-assistant
Azure Container App     — Live in Korea Central
GitHub Actions CI/CD    — Auto test + build on push


---

## Quick Start

### Run locally
```bash
git clone https://github.com/sricharanchavali37/smart-retail-assistant.git
cd smart-retail-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python backend/ml/run_phase1.py
uvicorn backend.main:app --port 9000
```

### Run with Docker
```bash
docker pull sricharanchavali37/smart-retail-assistant:latest
docker run -p 9000:9000 sricharanchavali37/smart-retail-assistant:latest
```

---

## ML Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost Demand Forecast | MAPE | 8.87% |
| XGBoost Demand Forecast | MAE | 28.30 |
| Isolation Forest | Anomalies Detected | 53 (5.02%) |
| Isolation Forest | Top Anomaly Z-Score | 6.3 |

---

## AI Agents

| Agent | Trigger Keywords | Capability |
|-------|-----------------|------------|
| Retail Analyst | revenue, highest, compare, 2023 | Analyses sales data with GPT-4o |
| Product Knowledge | return policy, what is, tell me | RAG over knowledge documents |
| Forecast Insight | forecast, predict, stock up | XGBoost + GPT-4o recommendations |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /ingest/ | Upload retail CSV |
| GET | /predict/ | Demand forecast by category |
| POST | /search/ | Search knowledge base |
| POST | /agent/ | Chat with AI agents |

---

## Azure Services

- **Azure OpenAI** — GPT-4o deployment
- **Azure Blob Storage** — smartretailorg, bronze container
- **Azure Data Factory** — Medallion pipeline (bronze to silver to gold)
- **Azure Container Apps** — Live deployment Korea Central

---

## Tests

```bash
pytest backend/tests/ -v
# 18 tests: ingest, predict, agent endpoints
```

---

## Project Structure

smart-retail-assistant/
├── backend/
│   ├── agents/      LangChain AI agents + RAG
│   ├── db/          Azure SQL + Blob clients
│   ├── ml/          XGBoost + Isolation Forest
│   ├── routes/      FastAPI route handlers
│   └── tests/       18 pytest unit tests
├── data/raw/        retail_sales.csv
├── notebooks/       Databricks PySpark notebooks
├── adf/             Azure Data Factory pipeline
├── powerbi/         Power BI dashboard
├── scripts/         Azure Blob upload
├── Dockerfile       Container definition
└── requirements.txt Python dependencies

---

## Author

**CH. Raja Sricharan** — Capstone Project 2025


