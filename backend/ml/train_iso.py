"""
train_iso.py
────────────
Trains Isolation Forest to detect anomalous sales days per category:
  • Sudden revenue spikes (viral product, pricing error)
  • Unusual quantity orders
  • Revenue drops (stock-out, data quality issue)

Outputs:
  backend/models/iso_anomaly.pkl     ← trained Isolation Forest
  backend/models/iso_scaler.pkl      ← StandardScaler for inference
  backend/models/iso_features.pkl    ← feature list for API inference
  data/processed/anomaly_scores.csv  ← all rows with anomaly flag + score

Run:
    python backend/ml/train_iso.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import (
    PROCESSED_CSV,
    ISO_MODEL_PATH, ISO_SCALER_PATH, ISO_FEATURES_PATH,
    ISO_PARAMS, ANOMALY_CSV,
    MODELS_DIR, DATA_PROCESSED_DIR,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Features for anomaly detection — focused on deviation from expected
# ─────────────────────────────────────────────────────────────────────────────
ISO_FEATURES = [
    "daily_revenue",
    "daily_quantity",
    "transaction_count",
    "revenue_z_score",           # std deviations from category mean
    "pct_from_rolling_mean",     # % change from recent 7-day average
    "rolling_mean_7",            # recent baseline revenue
    "rolling_std_7",             # recent revenue volatility
    "rolling_max_7",             # recent peak
    "rolling_min_7",             # recent trough
    "avg_price",                 # unusually high/low avg price = anomaly
    "avg_age",                   # unusual customer demographics
    "female_ratio",
    "DayOfWeek",                 # weekday pattern matters
    "Month",                     # seasonality
    "IsWeekend",
    "category_encoded",          # different normal ranges per category
]


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────
def analyze_anomalies(df: pd.DataFrame) -> None:
    log.info("\n  ── Anomaly Summary ──────────────────────────────────────")
    total    = len(df)
    n_anom   = int(df["anomaly_flag"].sum())
    pct      = n_anom / total * 100
    log.info(f"  Total records  : {total:,}")
    log.info(f"  Anomalies (1)  : {n_anom}  ({pct:.2f}%)")
    log.info(f"  Normal (0)     : {total - n_anom}  ({100 - pct:.2f}%)")

    log.info("\n  ── Anomaly Score Percentiles ────────────────────────────")
    for p in [50, 75, 90, 95, 99]:
        v = np.percentile(df["anomaly_score"], p)
        log.info(f"    P{p:02d} : {v:.4f}")

    log.info("\n  ── Top 10 Most Anomalous Records ───────────────────────")
    top_cols = ["Date", "Product Category", "daily_revenue", "daily_quantity",
                "transaction_count", "revenue_z_score", "anomaly_score"]
    top = df.nlargest(10, "anomaly_score")[top_cols].reset_index(drop=True)
    log.info("\n" + top.to_string(index=True))

    log.info("\n  ── Anomalies by Product Category ───────────────────────")
    by_cat = (
        df[df["anomaly_flag"] == 1]
        .groupby("Product Category")["anomaly_flag"]
        .count()
        .rename("anomaly_count")
    )
    log.info("\n" + by_cat.to_string())

    log.info("\n  ── Anomalies by Month ───────────────────────────────────")
    by_month = (
        df[df["anomaly_flag"] == 1]
        .groupby("Month")["anomaly_flag"]
        .count()
        .rename("anomaly_count")
    )
    log.info("\n" + by_month.to_string())

    log.info("\n  ── Anomalies on Weekends vs Weekdays ────────────────────")
    by_weekend = (
        df[df["anomaly_flag"] == 1]
        .groupby("IsWeekend")["anomaly_flag"]
        .count()
        .rename({0: "Weekday", 1: "Weekend"})
    )
    log.info("\n" + by_weekend.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train() -> None:
    log.info("=" * 58)
    log.info("  ISOLATION FOREST ANOMALY DETECTION TRAINING")
    log.info("  Target: anomalous daily revenue per category")
    log.info("=" * 58)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"  ❌  {PROCESSED_CSV} not found.\n"
            "  Run feature_eng.py first:\n"
            "      python backend/ml/feature_eng.py"
        )
    log.info(f"Loading {PROCESSED_CSV} …")
    df = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
    log.info(f"  Shape: {df.shape}")

    # ── Validate features ────────────────────────────────────────
    missing = [f for f in ISO_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing Isolation Forest features: {missing}")

    X = df[ISO_FEATURES].copy()

    # ── Scale ────────────────────────────────────────────────────
    log.info("\nFitting StandardScaler …")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info(f"  Scaled shape: {X_scaled.shape}")

    # ── Train ────────────────────────────────────────────────────
    log.info(
        f"\nTraining Isolation Forest  "
        f"(n_estimators={ISO_PARAMS['n_estimators']}, "
        f"contamination={ISO_PARAMS['contamination']}) …"
    )
    model = IsolationForest(**ISO_PARAMS)
    model.fit(X_scaled)

    # ── Predict ──────────────────────────────────────────────────
    log.info("Predicting anomalies …")
    raw_preds  = model.predict(X_scaled)     # 1=normal, -1=anomaly
    raw_scores = model.score_samples(X_scaled)

    # Flip: anomaly_flag=1 means anomaly; higher score = more anomalous
    df["anomaly_flag"]  = np.where(raw_preds == -1, 1, 0)
    df["anomaly_score"] = -raw_scores

    # ── Analyse ──────────────────────────────────────────────────
    analyze_anomalies(df)

    # ── Save model artefacts ─────────────────────────────────────
    joblib.dump(model,        ISO_MODEL_PATH)
    joblib.dump(scaler,       ISO_SCALER_PATH)
    joblib.dump(ISO_FEATURES, ISO_FEATURES_PATH)
    log.info(f"\n  Model saved    → {ISO_MODEL_PATH}")
    log.info(f"  Scaler saved   → {ISO_SCALER_PATH}")
    log.info(f"  Features saved → {ISO_FEATURES_PATH}")

    # ── Save anomaly CSV (feeds Power BI anomaly alerts page) ────
    output_cols = [
        "Date", "Product Category",
        "daily_revenue", "daily_quantity", "transaction_count",
        "avg_price", "avg_age", "female_ratio",
        "DayOfWeek", "Month", "IsWeekend",
        "rolling_mean_7", "rolling_std_7",
        "revenue_z_score", "pct_from_rolling_mean",
        "anomaly_flag", "anomaly_score",
    ]
    df[output_cols].to_csv(ANOMALY_CSV, index=False)
    log.info(f"  Anomaly CSV    → {ANOMALY_CSV}")

    # ── Sanity check ─────────────────────────────────────────────
    log.info("\nSanity check — reload and predict one row …")
    loaded_model  = joblib.load(ISO_MODEL_PATH)
    loaded_scaler = joblib.load(ISO_SCALER_PATH)

    sample_raw    = X.iloc[:1]
    sample_scaled = loaded_scaler.transform(sample_raw)
    pred          = loaded_model.predict(sample_scaled)[0]
    score         = -loaded_model.score_samples(sample_scaled)[0]
    label         = "ANOMALY ⚠️" if pred == -1 else "Normal ✅"
    log.info(f"  Row 0 → {label}  (score={score:.4f})")
    log.info(f"  revenue={sample_raw['daily_revenue'].iloc[0]:,.0f}  "
             f"z={sample_raw['revenue_z_score'].iloc[0]:.2f}")

    log.info("\n✅  train_iso.py complete.")


if __name__ == "__main__":
    train()