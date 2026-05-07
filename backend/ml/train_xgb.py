"""
train_xgb.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    PROCESSED_CSV,
    XGB_MODEL_PATH, XGB_FEATURES_PATH,
    XGB_PARAMS, TEST_WEEKS,
    MODELS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

FEATURES = [
    "category_encoded",
    "is_beauty", "is_clothing", "is_electronics",
    "Year", "Month", "Day", "DayOfWeek", "WeekOfYear",
    "DayOfYear", "Quarter", "IsWeekend", "IsMonthStart", "IsMonthEnd",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14",
    "rolling_std_7",  "rolling_std_14",
    "rolling_max_7",  "rolling_min_7",
    "avg_age", "female_ratio",
    "transaction_count", "avg_price",
    "revenue_z_score", "pct_from_rolling_mean",
]

TARGET = "daily_revenue"


def time_split(df, test_weeks=TEST_WEEKS):
    cutoff = df["Date"].max() - pd.Timedelta(weeks=test_weeks)
    train  = df[df["Date"] <= cutoff].copy()
    val    = df[df["Date"] >  cutoff].copy()
    log.info(f"  Cutoff  : {cutoff.date()}")
    log.info(f"  Train   : {len(train):,} rows  ({train['Date'].min().date()} -> {train['Date'].max().date()})")
    log.info(f"  Val     : {len(val):,} rows  ({val['Date'].min().date()} -> {val['Date'].max().date()})")
    return train, val


def mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred, split):
    mae_val  = mean_absolute_error(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true.values, y_pred)
    log.info(f"\n  -- {split} Metrics --")
    log.info(f"     MAE  = {mae_val:>10,.2f}")
    log.info(f"     RMSE = {rmse_val:>10,.2f}")
    log.info(f"     MAPE = {mape_val:>9.2f}%")
    return {"split": split, "mae": mae_val, "rmse": rmse_val, "mape": mape_val}


def train():
    log.info("=" * 58)
    log.info("  XGBOOST DEMAND (REVENUE) FORECASTING TRAINING")
    log.info("  Target: daily_revenue per product category")
    log.info("=" * 58)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(
            f"  {PROCESSED_CSV} not found.\n"
            "  Run feature_eng.py first:\n"
            "      python backend/ml/feature_eng.py"
        )

    log.info(f"Loading {PROCESSED_CSV} ...")
    df = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
    log.info(f"  Shape: {df.shape}")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns (re-run feature_eng.py): {missing}")

    log.info("\nSplitting data ...")
    train_df, val_df = time_split(df)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_val   = val_df[FEATURES]
    y_val   = val_df[TARGET]

    log.info("\n  -- Revenue per Category (train split) --")
    for cat in df["Product Category"].unique():
        cat_rev = train_df[train_df["Product Category"] == cat]["daily_revenue"]
        log.info(f"     {cat:<12}  mean={cat_rev.mean():,.0f}  max={cat_rev.max():,.0f}  min={cat_rev.min():,.0f}")

    log.info(f"\nTraining XGBoost ...  (n_estimators={XGB_PARAMS['n_estimators']}, max_depth={XGB_PARAMS['max_depth']}, lr={XGB_PARAMS['learning_rate']})\n")

    model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=30)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    log.info(f"\n  Best iteration: {model.best_iteration}")

    log.info("\nEvaluating ...")
    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_val)

    evaluate(y_train, train_preds, "TRAIN")
    val_metrics = evaluate(y_val, val_preds, "VALIDATION")

    log.info("\n  -- Validation MAPE by Category --")
    val_df = val_df.copy()
    val_df["pred"] = val_preds
    for cat in val_df["Product Category"].unique():
        mask = val_df["Product Category"] == cat
        y_c  = val_df.loc[mask, TARGET].values
        p_c  = val_df.loc[mask, "pred"].values
        m    = mape(y_c, p_c)
        log.info(f"     {cat:<14}  MAPE={m:.2f}%")

    imp = pd.Series(model.feature_importances_, index=FEATURES)
    log.info("\n  -- Top 10 Feature Importances --")
    log.info(imp.nlargest(10).to_string())

    joblib.dump(model,    XGB_MODEL_PATH)
    joblib.dump(FEATURES, XGB_FEATURES_PATH)
    log.info(f"\n  Model saved    -> {XGB_MODEL_PATH}")
    log.info(f"  Features saved -> {XGB_FEATURES_PATH}")

    log.info("\nSanity check -- reload and predict ...")
    loaded   = joblib.load(XGB_MODEL_PATH)
    s_X      = X_val.iloc[:1]
    s_y      = float(y_val.iloc[0])
    s_pred   = float(loaded.predict(s_X)[0])
    s_err    = abs(s_pred - s_y) / max(s_y, 1) * 100
    cat_name = val_df["Product Category"].iloc[0]
    log.info(f"  Category: {cat_name}")
    log.info(f"  Actual  : {s_y:,.0f}")
    log.info(f"  Predict : {s_pred:,.0f}")
    log.info(f"  Error   : {s_err:.1f}%")

    log.info("\n  train_xgb.py complete.")
    log.info(f"    Val MAE={val_metrics['mae']:,.0f}  RMSE={val_metrics['rmse']:,.0f}  MAPE={val_metrics['mape']:.2f}%")


if __name__ == "__main__":
    train()