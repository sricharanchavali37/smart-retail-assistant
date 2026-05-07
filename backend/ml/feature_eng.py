"""
feature_eng.py
──────────────
Loads retail_sales.csv (1000 transactions, 2023), aggregates to
daily level per Product Category, then engineers:

  • Aggregation    : daily revenue, quantity, transaction count per category
  • Date features  : year, month, day, weekday, quarter, weekend flag,
                     month-start/end flags, day of year
  • Lag features   : lag_1, lag_7, lag_14  (lag_365 skipped — only 1 year)
  • Rolling stats  : mean / std / max / min for 7 and 14-day windows
  • Category enc.  : one-hot + ordinal for Product Category
  • Customer stats : avg age, gender ratio per category per day
  • Deviation cols : z-score, % from rolling mean (for Isolation Forest)

Output: data/processed/retail_features.csv

Run from project root:
    python backend/ml/feature_eng.py
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import (
    RETAIL_CSV, PROCESSED_CSV,
    DATA_PROCESSED_DIR, CATEGORIES,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load raw transactions
# ─────────────────────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    if not RETAIL_CSV.exists():
        raise FileNotFoundError(
            f"\n\n  ❌  {RETAIL_CSV} not found.\n"
            "  Steps to fix:\n"
            "    1. Rename your dataset file  →  retail_sales.csv\n"
            "    2. Place it in               →  data/raw/retail_sales.csv\n"
        )

    log.info(f"Loading {RETAIL_CSV} …")
    df = pd.read_csv(RETAIL_CSV, parse_dates=["Date"])

    # ── Strip markdown bold markers from column names if present ──────────
    # The file may have **Total Amount** as column header
    df.columns = [c.strip().strip("*").strip() for c in df.columns]

    log.info(f"  Raw shape : {df.shape}")
    log.info(f"  Columns   : {list(df.columns)}")
    log.info(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    log.info(f"  Categories: {df['Product Category'].unique().tolist()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Validate raw data
# ─────────────────────────────────────────────────────────────────────────────
def validate_raw(df: pd.DataFrame) -> None:
    required = ["Date", "Customer ID", "Gender", "Age",
                "Product Category", "Quantity", "Price per Unit", "Total Amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in retail_sales.csv: {missing}\n"
            "Check that the file is saved correctly."
        )

    assert df["Total Amount"].isna().sum() == 0, "NaN in Total Amount"
    assert df["Quantity"].isna().sum() == 0,     "NaN in Quantity"
    assert df["Date"].isna().sum() == 0,          "NaN in Date"
    log.info("  Raw validation passed ✅")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Aggregate to daily × category level
#    Each row = one category on one date
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Aggregating to daily × category level …")

    # ── Revenue, quantity, transaction count ────────────────────────────
    agg = (
        df.groupby(["Date", "Product Category"])
        .agg(
            daily_revenue    = ("Total Amount", "sum"),
            daily_quantity   = ("Quantity", "sum"),
            transaction_count= ("Transaction ID", "count"),
            avg_price        = ("Price per Unit", "mean"),
            avg_age          = ("Age", "mean"),
        )
        .reset_index()
    )

    # ── Gender ratio (female proportion per category per day) ───────────
    gender_ratio = (
        df.assign(is_female=(df["Gender"] == "Female").astype(int))
        .groupby(["Date", "Product Category"])["is_female"]
        .mean()
        .reset_index()
        .rename(columns={"is_female": "female_ratio"})
    )
    agg = agg.merge(gender_ratio, on=["Date", "Product Category"], how="left")

    # ── Fill missing category-date combinations with 0 ─────────────────
    # (if a category had 0 transactions on a day, it won't appear in agg)
    all_dates      = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    all_categories = pd.CategoricalIndex(CATEGORIES)
    full_idx       = pd.MultiIndex.from_product(
        [all_dates, all_categories], names=["Date", "Product Category"]
    )
    agg = (
        agg.set_index(["Date", "Product Category"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )
    # Restore datetime dtype after reindex
    agg["Date"] = pd.to_datetime(agg["Date"])

    agg = agg.sort_values(["Product Category", "Date"]).reset_index(drop=True)
    log.info(f"  After aggregation: {agg.shape}  "
             f"(~{agg['Date'].nunique()} days × {agg['Product Category'].nunique()} categories)")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 4. Date features
# ─────────────────────────────────────────────────────────────────────────────
def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering date features …")
    df["Year"]         = df["Date"].dt.year
    df["Month"]        = df["Date"].dt.month
    df["Day"]          = df["Date"].dt.day
    df["DayOfWeek"]    = df["Date"].dt.dayofweek   # 0=Monday … 6=Sunday
    df["WeekOfYear"]   = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"]    = df["Date"].dt.dayofyear
    df["Quarter"]      = df["Date"].dt.quarter
    df["IsWeekend"]    = (df["DayOfWeek"] >= 5).astype(int)
    df["IsMonthStart"] = (df["Day"] <= 5).astype(int)
    df["IsMonthEnd"]   = (df["Day"] >= 25).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Lag features — per category (shift within each category's time series)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering lag features …")
    df = df.sort_values(["Product Category", "Date"])

    # lag_365 skipped — only 1 year of data (would be 100% NaN)
    for lag in [1, 7, 14]:
        log.info(f"  lag_{lag} …")
        df[f"lag_{lag}"] = df.groupby("Product Category")["daily_revenue"].shift(lag)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Rolling statistics — per category
# ─────────────────────────────────────────────────────────────────────────────
def engineer_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering rolling features …")
    df = df.sort_values(["Product Category", "Date"])

    for window in [7, 14]:
        log.info(f"  rolling window={window} …")
        grp = df.groupby("Product Category")["daily_revenue"]

        df[f"rolling_mean_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )

    # Max / min over 7-day window
    grp = df.groupby("Product Category")["daily_revenue"]
    df["rolling_max_7"] = grp.transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).max()
    )
    df["rolling_min_7"] = grp.transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).min()
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Categorical encoding
# ─────────────────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Encoding categorical features …")

    # Ordinal encoding
    cat_map = {"Beauty": 0, "Clothing": 1, "Electronics": 2}
    df["category_encoded"] = df["Product Category"].map(cat_map).fillna(0).astype(int)

    # One-hot encoding (3 columns — useful for XGBoost)
    for cat in CATEGORIES:
        col_name = f"is_{cat.lower()}"
        df[col_name] = (df["Product Category"] == cat).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sales deviation (used by Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────
def add_sales_deviation(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Adding sales deviation columns …")

    cat_mean = df.groupby("Product Category")["daily_revenue"].transform("mean")
    cat_std  = df.groupby("Product Category")["daily_revenue"].transform("std").replace(0, 1)

    df["revenue_z_score"]       = (df["daily_revenue"] - cat_mean) / cat_std
    df["pct_from_rolling_mean"] = (
        (df["daily_revenue"] - df["rolling_mean_7"])
        / df["rolling_mean_7"].replace(0, 1)
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 9. Validate & save
# ─────────────────────────────────────────────────────────────────────────────
def validate_and_save(df: pd.DataFrame) -> None:
    log.info("\n─── Validation ──────────────────────────────────────")

    # After dropping NaN lags we expect at least ~300 rows
    # (365 days × 3 categories − first 14 days lag warmup)
    assert len(df) >= 300, f"Dataset too small after lag drops: {len(df)} rows"

    # All lag columns must be NaN-free after dropna step
    for col in ["lag_1", "lag_7", "lag_14"]:
        n = df[col].isna().sum()
        assert n == 0, f"NaN in {col}: {n} rows"

    # Revenue must be non-negative
    assert (df["daily_revenue"] >= 0).all(), "Negative daily_revenue found"

    log.info(f"  Rows          : {len(df):,}")
    log.info(f"  Categories    : {df['Product Category'].nunique()}")
    log.info(f"  Date range    : {df['Date'].min().date()} → {df['Date'].max().date()}")
    log.info(f"  Columns       : {len(df.columns)}")

    nan_check = df.isna().sum()
    nan_cols  = nan_check[nan_check > 0]
    if len(nan_cols) > 0:
        log.warning(f"  Remaining NaNs:\n{nan_cols}")
    else:
        log.info("  No NaNs remaining ✅")

    log.info(f"\n  Daily revenue stats:\n{df['daily_revenue'].describe().to_string()}")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    log.info(f"\n  Saved → {PROCESSED_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    log.info("=" * 58)
    log.info("  FEATURE ENGINEERING PIPELINE")
    log.info("  Dataset: Retail Sales 2023 (transaction-level)")
    log.info("=" * 58)

    df_raw = load_raw()
    validate_raw(df_raw)
    df = aggregate_daily(df_raw)
    df = engineer_date_features(df)
    df = engineer_lag_features(df)
    df = engineer_rolling_features(df)
    df = encode_categoricals(df)
    df = add_sales_deviation(df)

    # Drop rows where lag_14 is NaN (first 14 days per category)
    before = len(df)
    df = df.dropna(subset=["lag_1", "lag_7", "lag_14"]).reset_index(drop=True)
    log.info(f"\nDropped {before - len(df):,} rows with NaN lags (lag warmup period)")

    validate_and_save(df)
    log.info("\n✅  feature_eng.py complete.")
    return df


if __name__ == "__main__":
    run()