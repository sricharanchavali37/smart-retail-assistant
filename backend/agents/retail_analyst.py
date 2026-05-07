"""
retail_analyst.py
─────────────────
Agent 1: Retail Analyst Agent
Analyses sales data, trends, anomalies from retail_features.csv
and anomaly_scores.csv using Azure OpenAI GPT-4o.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_CSV = ROOT / "data" / "processed" / "retail_features.csv"
ANOMALY_CSV   = ROOT / "data" / "processed" / "anomaly_scores.csv"

SYSTEM_PROMPT = """You are a Retail Analyst AI for a Smart Retail store.
You have access to sales data for Beauty, Clothing, and Electronics categories.
Analyse trends, compare categories, identify anomalies, and give actionable insights.
Always be specific — mention category names, revenue figures, and dates when relevant.
Keep answers concise and business-focused."""


def _get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version="2024-02-01",
        temperature=0.3,
        max_tokens=800,
    )


def _build_data_context() -> str:
    context_parts = []

    if PROCESSED_CSV.exists():
        df      = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
        df_2023 = df[df["Date"].dt.year == 2023]
        summary = df_2023.groupby("Product Category")["daily_revenue"].agg(
            ["sum", "mean", "max", "min"]
        ).round(2)
        context_parts.append("=== SALES SUMMARY (2023) ===")
        context_parts.append(summary.to_string())

        monthly = df_2023.groupby(
            [df_2023["Date"].dt.month_name(), "Product Category"]
        )["daily_revenue"].sum().unstack(fill_value=0)
        context_parts.append("\n=== MONTHLY REVENUE BY CATEGORY ===")
        context_parts.append(monthly.to_string())

    if ANOMALY_CSV.exists():
        adf      = pd.read_csv(ANOMALY_CSV, parse_dates=["Date"])
        anomalies = adf[adf["anomaly_flag"] == 1].sort_values(
            "anomaly_score", ascending=False
        ).head(10)
        context_parts.append("\n=== TOP 10 ANOMALOUS DAYS ===")
        context_parts.append(
            anomalies[["Date", "Product Category", "daily_revenue",
                        "revenue_z_score", "anomaly_score"]].to_string(index=False)
        )
        by_cat = adf[adf["anomaly_flag"] == 1].groupby(
            "Product Category"
        )["anomaly_flag"].count()
        context_parts.append("\n=== ANOMALY COUNT BY CATEGORY ===")
        context_parts.append(by_cat.to_string())

    return "\n".join(context_parts)


def run(query: str, session_id: str = "") -> dict:
    log.info(f"[RetailAnalystAgent] query='{query[:80]}'")
    data_context = _build_data_context()
    full_system  = f"{SYSTEM_PROMPT}\n\nRetail data:\n\n{data_context}"

    try:
        llm      = _get_llm()
        messages = [
            SystemMessage(content=full_system),
            HumanMessage(content=query),
        ]
        response = llm.invoke(messages)
        return {"agent": "retail_analyst", "response": response.content, "status": "success"}
    except Exception as e:
        log.error(f"[RetailAnalystAgent] failed: {e}")
        return {"agent": "retail_analyst", "response": f"Error: {str(e)}", "status": "error"}