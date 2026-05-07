"""
config.py
─────────
Single source of truth for all paths and hyperparameters used across
feature engineering, XGBoost training, and Isolation Forest training.

Dataset: Retail Sales Dataset (1000 transactions, 2023, 3 categories)
Columns: Transaction ID, Date, Customer ID, Gender, Age,
         Product Category, Quantity, Price per Unit, Total Amount

Do NOT hardcode paths in any other file — always import from here.
"""

from pathlib import Path

# ── Root resolution ─────────────────────────────────────────────────────────
# __file__ = D:\projects\sowmya\backend\ml\config.py
# parents[0] = backend\ml
# parents[1] = backend
# parents[2] = D:\projects\sowmya  ← project root
ROOT_DIR           = Path(__file__).resolve().parents[2]

# ── Data directories ────────────────────────────────────────────────────────
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ── Raw file (rename your dataset to this before running) ───────────────────
RETAIL_CSV         = DATA_RAW_DIR / "retail_sales.csv"

# ── Processed outputs ───────────────────────────────────────────────────────
PROCESSED_CSV      = DATA_PROCESSED_DIR / "retail_features.csv"
ANOMALY_CSV        = DATA_PROCESSED_DIR / "anomaly_scores.csv"

# ── Model artifacts ──────────────────────────────────────────────────────────
MODELS_DIR         = ROOT_DIR / "backend" / "models"
XGB_MODEL_PATH     = MODELS_DIR / "xgb_demand.pkl"
XGB_FEATURES_PATH  = MODELS_DIR / "xgb_features.pkl"
ISO_MODEL_PATH     = MODELS_DIR / "iso_anomaly.pkl"
ISO_SCALER_PATH    = MODELS_DIR / "iso_scaler.pkl"
ISO_FEATURES_PATH  = MODELS_DIR / "iso_features.pkl"

# ── Product categories in the dataset ────────────────────────────────────────
CATEGORIES         = ["Beauty", "Clothing", "Electronics"]

# ── XGBoost hyperparameters ──────────────────────────────────────────────────
# Tuned for small dataset (~1000 rows after aggregation)
# Lower n_estimators + higher learning_rate prevents overfitting on small data
XGB_PARAMS = {
    "n_estimators"     : 300,
    "max_depth"        : 4,          # shallow — avoids overfit on small data
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "min_child_weight" : 2,
    "reg_alpha"        : 0.5,        # stronger L1 for small dataset
    "reg_lambda"       : 1.5,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "tree_method"      : "hist",
    "eval_metric"      : "mae",
}

# Hold out last N weeks as validation
# 4 weeks = ~28 days — appropriate for 1 year of data
TEST_WEEKS = 4

# ── Isolation Forest hyperparameters ─────────────────────────────────────────
ISO_PARAMS = {
    "contamination" : 0.05,   # flag ~5% of days as anomalous
    "n_estimators"  : 100,
    "max_samples"   : "auto",
    "random_state"  : 42,
    "n_jobs"        : -1,
}