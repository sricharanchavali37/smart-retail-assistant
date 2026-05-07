"""
verify_phase1.py
────────────────
Run after Phase 1 to confirm every output file exists, loads cleanly,
and produces a sane prediction before you start Phase 2.

Run from project root:
    python backend/ml/verify_phase1.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "✅" if condition else "❌"
    msg    = f"  {status}  {label}"
    if detail:
        msg += f"  →  {detail}"
    log.info(msg)
    return condition


def main() -> None:
    import joblib
    import numpy as np
    import pandas as pd

    from config import (
        PROCESSED_CSV, ANOMALY_CSV,
        XGB_MODEL_PATH, XGB_FEATURES_PATH,
        ISO_MODEL_PATH, ISO_SCALER_PATH, ISO_FEATURES_PATH,
    )

    log.info("=" * 55)
    log.info("  PHASE 1 VERIFICATION")
    log.info("=" * 55)

    failures = 0

    # ── File existence ────────────────────────────────────────────
    log.info("\n── File existence ──────────────────────────────────")
    for path in [
        PROCESSED_CSV, ANOMALY_CSV,
        XGB_MODEL_PATH, XGB_FEATURES_PATH,
        ISO_MODEL_PATH, ISO_SCALER_PATH, ISO_FEATURES_PATH,
    ]:
        exists = path.exists()
        size   = path.stat().st_size / 1024 if exists else 0
        if not check(path.name, exists, f"{size:,.0f} KB"):
            failures += 1

    # ── Processed CSV ─────────────────────────────────────────────
    log.info("\n── Processed CSV ───────────────────────────────────")
    try:
        df = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
        check("Rows >= 300",               len(df) >= 300,     f"{len(df):,} rows")
        check("Columns >= 20",             len(df.columns) >= 20, f"{len(df.columns)} cols")
        check("No NaN in daily_revenue",   df["daily_revenue"].isna().sum() == 0)
        check("No NaN in lag_1",           df["lag_1"].isna().sum() == 0)
        check("No NaN in lag_7",           df["lag_7"].isna().sum() == 0)
        check("No NaN in lag_14",          df["lag_14"].isna().sum() == 0)
        check("rolling_mean_7 present",    "rolling_mean_7" in df.columns)
        check("revenue_z_score present",   "revenue_z_score" in df.columns)
        check("category_encoded present",  "category_encoded" in df.columns)
        check("3 categories present",
              df["Product Category"].nunique() == 3,
              str(df["Product Category"].unique().tolist()))
        check("daily_revenue >= 0",        (df["daily_revenue"] >= 0).all())

        log.info(f"  Date range : {df['Date'].min().date()} → {df['Date'].max().date()}")
        log.info(f"  Categories : {df['Product Category'].unique().tolist()}")
        log.info(f"  Revenue stats:\n{df['daily_revenue'].describe().to_string()}")

    except Exception as e:
        log.error(f"  ❌  Processed CSV verification failed: {e}")
        failures += 1

    # ── XGBoost model ─────────────────────────────────────────────
    log.info("\n── XGBoost Model ───────────────────────────────────")
    try:
        model    = joblib.load(XGB_MODEL_PATH)
        features = joblib.load(XGB_FEATURES_PATH)

        check("XGBRegressor loaded",   True,               type(model).__name__)
        check("Feature list loaded",   True,               f"{len(features)} features")
        check("≥ 20 features",         len(features) >= 20, f"{len(features)}")

        # Synthetic sample — Electronics on a Monday in June
        sample = pd.DataFrame([{f: 0 for f in features}])
        sample["category_encoded"]    = 2   # Electronics
        sample["is_electronics"]      = 1
        sample["DayOfWeek"]           = 0   # Monday
        sample["Month"]               = 6
        sample["lag_1"]               = 800
        sample["lag_7"]               = 750
        sample["lag_14"]              = 720
        sample["rolling_mean_7"]      = 760
        sample["avg_price"]           = 200
        sample["transaction_count"]   = 5

        pred = float(model.predict(sample)[0])
        check("Prediction > 0",  pred > 0, f"predicted=₹{pred:,.0f}")

    except Exception as e:
        log.error(f"  ❌  XGBoost verification failed: {e}")
        failures += 1

    # ── Isolation Forest model ────────────────────────────────────
    log.info("\n── Isolation Forest Model ──────────────────────────")
    try:
        iso_model    = joblib.load(ISO_MODEL_PATH)
        iso_scaler   = joblib.load(ISO_SCALER_PATH)
        iso_features = joblib.load(ISO_FEATURES_PATH)

        check("IsolationForest loaded", True, type(iso_model).__name__)
        check("Scaler loaded",          True, type(iso_scaler).__name__)
        check("ISO feature list",       len(iso_features) > 0, f"{len(iso_features)} features")

        sample_iso = np.zeros((1, len(iso_features)))
        scaled     = iso_scaler.transform(sample_iso)
        pred       = iso_model.predict(scaled)[0]
        check("ISO prediction returned", pred in [1, -1], f"pred={pred}")

    except Exception as e:
        log.error(f"  ❌  Isolation Forest verification failed: {e}")
        failures += 1

    # ── Anomaly CSV ───────────────────────────────────────────────
    log.info("\n── Anomaly Scores CSV ──────────────────────────────")
    try:
        adf = pd.read_csv(ANOMALY_CSV, parse_dates=["Date"])
        check("Rows >= 300",                len(adf) >= 300,       f"{len(adf):,}")
        check("anomaly_flag exists",        "anomaly_flag"  in adf.columns)
        check("anomaly_score exists",       "anomaly_score" in adf.columns)
        n_anom = int(adf["anomaly_flag"].sum())
        pct    = n_anom / len(adf) * 100
        check("Has some anomalies",         n_anom > 0,            f"{n_anom} flagged ({pct:.1f}%)")
        check("Anomaly % between 1–15%",    1 <= pct <= 15,        f"{pct:.2f}%")
    except Exception as e:
        log.error(f"  ❌  Anomaly CSV verification failed: {e}")
        failures += 1

    # ── Final verdict ─────────────────────────────────────────────
    log.info("\n" + "═" * 55)
    if failures == 0:
        log.info("  ALL CHECKS PASSED ✅  Ready for Phase 2.")
        log.info("  Run: python backend/main.py  (after Phase 2 files are added)")
    else:
        log.info(f"  {failures} CHECK(S) FAILED ❌  Fix above errors before Phase 2.")
    log.info("═" * 55)


if __name__ == "__main__":
    main()