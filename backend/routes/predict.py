"""
predict.py
──────────
GET /predict
Loads xgb_demand.pkl and returns a demand revenue forecast
for a given product category over the next N days.

Query params:
  category   : str  - "Beauty" | "Clothing" | "Electronics"
  days_ahead : int  - 1 to 14  (default 7)

Response: JSON with daily forecast + anomaly flags from Isolation Forest
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

log = logging.getLogger(__name__)
router = APIRouter()

# ── Model paths ───────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
MODELS_DIR    = ROOT / "backend" / "models"
PROCESSED_CSV = ROOT / "data" / "processed" / "retail_features.csv"
ANOMALY_CSV   = ROOT / "data" / "processed" / "anomaly_scores.csv"

VALID_CATEGORIES = ["Beauty", "Clothing", "Electronics"]
CATEGORY_MAP     = {"Beauty": 0, "Clothing": 1, "Electronics": 2}

# ── Load models once at import time ──────────────────────────────────────────
def _load_model(path: Path, label: str):
    if not path.exists():
        log.warning(f"{label} not found at {path} — run Phase 1 training first.")
        return None
    try:
        obj = joblib.load(path)
        log.info(f"Loaded {label} from {path}")
        return obj
    except Exception as e:
        log.error(f"Failed to load {label}: {e}")
        return None


xgb_model    = _load_model(MODELS_DIR / "xgb_demand.pkl",   "XGBoost model")
xgb_features = _load_model(MODELS_DIR / "xgb_features.pkl", "XGBoost features")
iso_model    = _load_model(MODELS_DIR / "iso_anomaly.pkl",  "Isolation Forest")
iso_scaler   = _load_model(MODELS_DIR / "iso_scaler.pkl",   "ISO scaler")
iso_features = _load_model(MODELS_DIR / "iso_features.pkl", "ISO features")


def _get_latest_actuals(category: str) -> dict:
    """
    Pull the most recent known values for a category from retail_features.csv.
    Used to seed lag and rolling features for future date predictions.
    """
    if not PROCESSED_CSV.exists():
        return {}
    try:
        df = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
        cat_df = df[df["Product Category"] == category].sort_values("Date")
        if cat_df.empty:
            return {}
        last = cat_df.iloc[-1]
        return {
            "last_date":         last["Date"],
            "lag_1":             last["daily_revenue"],
            "lag_7":             cat_df.iloc[-7]["daily_revenue"]  if len(cat_df) >= 7  else last["daily_revenue"],
            "lag_14":            cat_df.iloc[-14]["daily_revenue"] if len(cat_df) >= 14 else last["daily_revenue"],
            "rolling_mean_7":    last["rolling_mean_7"],
            "rolling_mean_14":   last["rolling_mean_14"],
            "rolling_std_7":     last["rolling_std_7"],
            "rolling_std_14":    last["rolling_std_14"],
            "rolling_max_7":     last["rolling_max_7"],
            "rolling_min_7":     last["rolling_min_7"],
            "avg_age":           last.get("avg_age", 35.0),
            "female_ratio":      last.get("female_ratio", 0.5),
            "transaction_count": last.get("transaction_count", 3),
            "avg_price":         last.get("avg_price", 150.0),
            "revenue_z_score":   0.0,
            "pct_from_rolling_mean": 0.0,
        }
    except Exception as e:
        log.error(f"_get_latest_actuals failed: {e}")
        return {}


def _build_feature_row(target_date: datetime, category: str, actuals: dict) -> pd.DataFrame:
    """Build a single feature row for one future date."""
    cat_enc = CATEGORY_MAP.get(category, 0)
    row = {
        "category_encoded":      cat_enc,
        "is_beauty":             1 if category == "Beauty"      else 0,
        "is_clothing":           1 if category == "Clothing"    else 0,
        "is_electronics":        1 if category == "Electronics" else 0,
        "Year":                  target_date.year,
        "Month":                 target_date.month,
        "Day":                   target_date.day,
        "DayOfWeek":             target_date.weekday(),
        "WeekOfYear":            target_date.isocalendar()[1],
        "DayOfYear":             target_date.timetuple().tm_yday,
        "Quarter":               (target_date.month - 1) // 3 + 1,
        "IsWeekend":             1 if target_date.weekday() >= 5 else 0,
        "IsMonthStart":          1 if target_date.day <= 5      else 0,
        "IsMonthEnd":            1 if target_date.day >= 25     else 0,
        "lag_1":                 actuals.get("lag_1",           300),
        "lag_7":                 actuals.get("lag_7",           300),
        "lag_14":                actuals.get("lag_14",          300),
        "rolling_mean_7":        actuals.get("rolling_mean_7",  300),
        "rolling_mean_14":       actuals.get("rolling_mean_14", 300),
        "rolling_std_7":         actuals.get("rolling_std_7",   100),
        "rolling_std_14":        actuals.get("rolling_std_14",  100),
        "rolling_max_7":         actuals.get("rolling_max_7",   500),
        "rolling_min_7":         actuals.get("rolling_min_7",   100),
        "avg_age":               actuals.get("avg_age",         35.0),
        "female_ratio":          actuals.get("female_ratio",    0.5),
        "transaction_count":     actuals.get("transaction_count", 3),
        "avg_price":             actuals.get("avg_price",       150.0),
        "revenue_z_score":       actuals.get("revenue_z_score", 0.0),
        "pct_from_rolling_mean": actuals.get("pct_from_rolling_mean", 0.0),
    }
    return pd.DataFrame([row])


def _predict_anomaly(feature_row: pd.DataFrame) -> dict:
    """Run Isolation Forest on a feature row and return anomaly info."""
    if iso_model is None or iso_scaler is None or iso_features is None:
        return {"anomaly_flag": 0, "anomaly_score": None}
    try:
        X = feature_row[iso_features].values
        X_scaled = iso_scaler.transform(X)
        flag  = int(iso_model.predict(X_scaled)[0])
        score = float(-iso_model.score_samples(X_scaled)[0])
        return {
            "anomaly_flag":  1 if flag == -1 else 0,
            "anomaly_score": round(score, 4),
        }
    except Exception as e:
        log.warning(f"Anomaly prediction failed: {e}")
        return {"anomaly_flag": 0, "anomaly_score": None}


@router.get(
    "/",
    summary="Demand forecast by category",
    description=(
        "Returns daily revenue forecast for a product category "
        "over the next N days using the trained XGBoost model."
    ),
)
async def predict(
    category:   str = Query(...,  description="Product category: Beauty | Clothing | Electronics"),
    days_ahead: int = Query(7,    description="Number of days to forecast (1–14)", ge=1, le=14),
):
    log.info(f"GET /predict  category={category}  days_ahead={days_ahead}")

    # ── Validate ─────────────────────────────────────────────────
    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Choose from: {VALID_CATEGORIES}",
        )

    if xgb_model is None or xgb_features is None:
        raise HTTPException(
            status_code=503,
            detail="XGBoost model not loaded. Run Phase 1 training first: python backend/ml/run_phase1.py",
        )

    # ── Build forecasts ──────────────────────────────────────────
    actuals   = _get_latest_actuals(category)
    start_date = (actuals.get("last_date", pd.Timestamp.now()) + timedelta(days=1))
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.to_pydatetime()

    forecasts = []
    rolling_actuals = actuals.copy()

    for i in range(days_ahead):
        target_date  = start_date + timedelta(days=i)
        feature_row  = _build_feature_row(target_date, category, rolling_actuals)

        # XGBoost prediction
        pred_revenue = float(xgb_model.predict(feature_row[xgb_features])[0])
        pred_revenue = max(0.0, round(pred_revenue, 2))

        # Anomaly check on predicted values
        anomaly_info = _predict_anomaly(feature_row)

        forecasts.append({
            "date":            target_date.strftime("%Y-%m-%d"),
            "day_of_week":     target_date.strftime("%A"),
            "predicted_revenue": pred_revenue,
            "anomaly_flag":    anomaly_info["anomaly_flag"],
            "anomaly_score":   anomaly_info["anomaly_score"],
        })

        # Roll forward: use prediction as next day's lag_1
        rolling_actuals["lag_14"] = rolling_actuals.get("lag_7",  pred_revenue)
        rolling_actuals["lag_7"]  = rolling_actuals.get("lag_1",  pred_revenue)
        rolling_actuals["lag_1"]  = pred_revenue

    # ── Summary stats ────────────────────────────────────────────
    revenues  = [f["predicted_revenue"] for f in forecasts]
    anomalies = [f for f in forecasts if f["anomaly_flag"] == 1]

    return {
        "category":   category,
        "days_ahead": days_ahead,
        "forecast":   forecasts,
        "summary": {
            "total_predicted_revenue": round(sum(revenues), 2),
            "avg_daily_revenue":       round(sum(revenues) / len(revenues), 2),
            "max_day":                 max(forecasts, key=lambda x: x["predicted_revenue"]),
            "min_day":                 min(forecasts, key=lambda x: x["predicted_revenue"]),
            "anomaly_days_count":      len(anomalies),
            "anomaly_days":            [a["date"] for a in anomalies],
        },
        "model": "XGBoost (xgb_demand.pkl)",
    }