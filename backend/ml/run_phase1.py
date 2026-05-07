"""
run_phase1.py
─────────────
Runs the complete Phase 1 pipeline in the correct order:
  1. Feature engineering  → data/processed/retail_features.csv
  2. XGBoost training     → backend/models/xgb_demand.pkl
  3. Isolation Forest     → backend/models/iso_anomaly.pkl

Run from project root (D:\\projects\\sowmya):
    python backend/ml/run_phase1.py

Expected total time: ~2–4 minutes (small dataset, fast on i5-13500H)
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def banner(step: int, title: str) -> None:
    log.info("\n" + "═" * 60)
    log.info(f"  STEP {step}/3 : {title}")
    log.info("═" * 60)


def main() -> None:
    total_start = time.time()

    # ── Step 1 ───────────────────────────────────────────────────
    banner(1, "FEATURE ENGINEERING")
    t0 = time.time()
    import feature_eng
    feature_eng.run()
    log.info(f"  ⏱  Done in {(time.time() - t0):.1f} sec")

    # ── Step 2 ───────────────────────────────────────────────────
    banner(2, "XGBOOST DEMAND FORECASTING")
    t0 = time.time()
    import train_xgb
    train_xgb.train()
    log.info(f"  ⏱  Done in {(time.time() - t0):.1f} sec")

    # ── Step 3 ───────────────────────────────────────────────────
    banner(3, "ISOLATION FOREST ANOMALY DETECTION")
    t0 = time.time()
    import train_iso
    train_iso.train()
    log.info(f"  ⏱  Done in {(time.time() - t0):.1f} sec")

    # ── Summary ──────────────────────────────────────────────────
    total_sec = time.time() - total_start
    log.info("\n" + "═" * 60)
    log.info("  PHASE 1 COMPLETE ✅")
    log.info(f"  Total time: {total_sec:.0f} seconds ({total_sec/60:.1f} min)")
    log.info("═" * 60)

    from config import (
        PROCESSED_CSV, ANOMALY_CSV,
        XGB_MODEL_PATH, XGB_FEATURES_PATH,
        ISO_MODEL_PATH, ISO_SCALER_PATH, ISO_FEATURES_PATH,
    )
    log.info("\n  Output files:")
    for f in [PROCESSED_CSV, ANOMALY_CSV,
               XGB_MODEL_PATH, XGB_FEATURES_PATH,
               ISO_MODEL_PATH, ISO_SCALER_PATH, ISO_FEATURES_PATH]:
        p    = Path(f)
        exists = p.exists()
        size = p.stat().st_size / 1024 if exists else 0
        status = "✅" if exists else "❌ MISSING"
        log.info(f"    {status}  {p.name:<32} ({size:,.0f} KB)")

    log.info("\n  Run verify_phase1.py to confirm all outputs before Phase 2.")


if __name__ == "__main__":
    main()