"""
forecast_insight.py
───────────────────
Agent 3: Forecast Insight Agent
Directly imports prediction logic (no HTTP self-call).
Uses Azure OpenAI GPT-4o to generate business recommendations.
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parents[2]
MODELS_DIR    = ROOT / "backend" / "models"
PROCESSED_CSV = ROOT / "data" / "processed" / "retail_features.csv"
CATEGORY_MAP  = {"Beauty": 0, "Clothing": 1, "Electronics": 2}
VALID_CATS    = ["Beauty", "Clothing", "Electronics"]

SYSTEM_PROMPT = """You are a Demand Forecasting AI for a Smart Retail store.
You interpret XGBoost machine learning forecasts for Beauty, Clothing, and Electronics.
Based on forecast data, give clear business recommendations:
  - Should the store stock up or reduce inventory?
  - Are there any anomaly risks in the forecast?
  - Which category needs attention this week?
Be specific — mention predicted revenue figures, dates, and actionable steps.
Keep answers brief and practical for a store manager."""

FEATURES = [
    "category_encoded","is_beauty","is_clothing","is_electronics",
    "Year","Month","Day","DayOfWeek","WeekOfYear","DayOfYear",
    "Quarter","IsWeekend","IsMonthStart","IsMonthEnd",
    "lag_1","lag_7","lag_14",
    "rolling_mean_7","rolling_mean_14","rolling_std_7","rolling_std_14",
    "rolling_max_7","rolling_min_7",
    "avg_age","female_ratio","transaction_count","avg_price",
    "revenue_z_score","pct_from_rolling_mean",
]


def _get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version="2024-02-01",
        temperature=0.3,
        max_tokens=700,
    )


def _load_models():
    try:
        model    = joblib.load(MODELS_DIR / "xgb_demand.pkl")
        features = joblib.load(MODELS_DIR / "xgb_features.pkl")
        return model, features
    except Exception as e:
        log.error(f"Model load failed: {e}")
        return None, None


def _get_latest_actuals(category: str) -> dict:
    if not PROCESSED_CSV.exists():
        return {}
    try:
        df     = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
        cat_df = df[df["Product Category"] == category].sort_values("Date")
        if cat_df.empty:
            return {}
        last = cat_df.iloc[-1]
        return {
            "last_date":             last["Date"],
            "lag_1":                 last["daily_revenue"],
            "lag_7":                 cat_df.iloc[-7]["daily_revenue"]  if len(cat_df) >= 7  else last["daily_revenue"],
            "lag_14":                cat_df.iloc[-14]["daily_revenue"] if len(cat_df) >= 14 else last["daily_revenue"],
            "rolling_mean_7":        last["rolling_mean_7"],
            "rolling_mean_14":       last["rolling_mean_14"],
            "rolling_std_7":         last["rolling_std_7"],
            "rolling_std_14":        last["rolling_std_14"],
            "rolling_max_7":         last["rolling_max_7"],
            "rolling_min_7":         last["rolling_min_7"],
            "avg_age":               last.get("avg_age", 35.0),
            "female_ratio":          last.get("female_ratio", 0.5),
            "transaction_count":     last.get("transaction_count", 3),
            "avg_price":             last.get("avg_price", 150.0),
            "revenue_z_score":       0.0,
            "pct_from_rolling_mean": 0.0,
        }
    except Exception as e:
        log.error(f"_get_latest_actuals failed: {e}")
        return {}


def _predict_category(category: str, days: int, model, feat_list) -> str:
    actuals    = _get_latest_actuals(category)
    cat_enc    = CATEGORY_MAP.get(category, 0)
    start_date = actuals.get("last_date", pd.Timestamp("2024-01-01"))
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.to_pydatetime()

    lines    = [f"\n--- {category} (next {days} days) ---"]
    rolling  = actuals.copy()
    total    = 0.0

    for i in range(days):
        d = start_date + timedelta(days=i + 1)
        row = {
            "category_encoded": cat_enc,
            "is_beauty":        1 if category == "Beauty"      else 0,
            "is_clothing":      1 if category == "Clothing"    else 0,
            "is_electronics":   1 if category == "Electronics" else 0,
            "Year": d.year, "Month": d.month, "Day": d.day,
            "DayOfWeek": d.weekday(),
            "WeekOfYear": d.isocalendar()[1],
            "DayOfYear":  d.timetuple().tm_yday,
            "Quarter":    (d.month - 1) // 3 + 1,
            "IsWeekend":  1 if d.weekday() >= 5 else 0,
            "IsMonthStart": 1 if d.day <= 5  else 0,
            "IsMonthEnd":   1 if d.day >= 25 else 0,
            "lag_1":             rolling.get("lag_1",  300),
            "lag_7":             rolling.get("lag_7",  300),
            "lag_14":            rolling.get("lag_14", 300),
            "rolling_mean_7":    rolling.get("rolling_mean_7",  300),
            "rolling_mean_14":   rolling.get("rolling_mean_14", 300),
            "rolling_std_7":     rolling.get("rolling_std_7",   100),
            "rolling_std_14":    rolling.get("rolling_std_14",  100),
            "rolling_max_7":     rolling.get("rolling_max_7",   500),
            "rolling_min_7":     rolling.get("rolling_min_7",   100),
            "avg_age":           rolling.get("avg_age",          35.0),
            "female_ratio":      rolling.get("female_ratio",     0.5),
            "transaction_count": rolling.get("transaction_count", 3),
            "avg_price":         rolling.get("avg_price",        150.0),
            "revenue_z_score":   0.0,
            "pct_from_rolling_mean": 0.0,
        }
        X    = pd.DataFrame([row])
        pred = max(0.0, float(model.predict(X[feat_list])[0]))
        total += pred
        lines.append(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%A'):<9}): Rs {pred:,.0f}")
        rolling["lag_14"] = rolling.get("lag_7",  pred)
        rolling["lag_7"]  = rolling.get("lag_1",  pred)
        rolling["lag_1"]  = pred

    lines.append(f"  Total: Rs {total:,.0f}  |  Avg/day: Rs {total/days:,.0f}")
    return "\n".join(lines)


def _build_forecast_context(query: str) -> str:
    model, feat_list = _load_models()
    if model is None:
        return "Forecast models not available. Run Phase 1 training first."

    q    = query.lower()
    cats = (
        ["Beauty"]      if "beauty"      in q else
        ["Clothing"]    if "clothing"    in q else
        ["Electronics"] if "electronics" in q else
        VALID_CATS
    )
    days  = 14 if ("14" in q or "two week" in q) else 7
    parts = [f"=== DEMAND FORECAST (next {days} days) ==="]
    for cat in cats:
        parts.append(_predict_category(cat, days, model, feat_list))
    return "\n".join(parts)


def run(query: str, session_id: str = "") -> dict:
    log.info(f"[ForecastInsightAgent] query='{query[:80]}'")
    forecast_context = _build_forecast_context(query)
    full_system      = f"{SYSTEM_PROMPT}\n\nForecast data:\n\n{forecast_context}"

    try:
        llm      = _get_llm()
        messages = [SystemMessage(content=full_system), HumanMessage(content=query)]
        response = llm.invoke(messages)
        return {
            "agent":            "forecast_insight",
            "response":         response.content,
            "status":           "success",
            "forecast_context": forecast_context,
        }
    except Exception as e:
        log.error(f"[ForecastInsightAgent] failed: {e}")
        return {"agent": "forecast_insight", "response": f"Error: {str(e)}", "status": "error"}