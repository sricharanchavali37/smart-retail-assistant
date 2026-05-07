"""
ingest.py
─────────
POST /ingest
Accepts a CSV file upload of retail sales transactions.
Validates columns, saves locally, uploads to Azure Blob (if configured),
inserts into Azure SQL (if configured).

Request : multipart/form-data  with field "file" (CSV)
Response: JSON with row count, validation summary, storage status
"""

import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.db.azure_sql  import insert_transactions, init_tables
from backend.db.blob_client import upload_csv

log = logging.getLogger(__name__)
router = APIRouter()

REQUIRED_COLUMNS = {
    "Transaction ID", "Date", "Customer ID", "Gender",
    "Age", "Product Category", "Quantity",
    "Price per Unit", "Total Amount",
}

VALID_CATEGORIES = {"Beauty", "Clothing", "Electronics"}

# Local backup path for uploaded files
DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Validate uploaded CSV and return a summary dict."""
    errors   = []
    warnings = []

    # Strip bold markdown from column names if present
    df.columns = [c.strip().strip("*").strip() for c in df.columns]

    # Column check
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {sorted(missing_cols)}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "df": df}

    # Parse dates
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        errors.append("Date column could not be parsed. Expected format: YYYY-MM-DD")

    # Numeric checks
    for col in ["Age", "Quantity", "Price per Unit", "Total Amount"]:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna().sum()
            if non_numeric > 0:
                warnings.append(f"{col}: {non_numeric} non-numeric values found")

    # Category check
    if "Product Category" in df.columns:
        unknown = set(df["Product Category"].unique()) - VALID_CATEGORIES
        if unknown:
            warnings.append(f"Unknown categories found: {unknown}")

    # Null check
    null_counts = df[list(REQUIRED_COLUMNS & set(df.columns))].isna().sum()
    for col, cnt in null_counts[null_counts > 0].items():
        warnings.append(f"{col}: {cnt} null values")

    return {
        "valid":    len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
        "df":       df,
    }


@router.post(
    "/",
    summary="Upload retail sales CSV",
    description="Upload a CSV of sales transactions. Validates, stores to Azure Blob + SQL.",
)
async def ingest(file: UploadFile = File(...)):
    log.info(f"POST /ingest  filename={file.filename}  content_type={file.content_type}")

    # ── Accept only CSV ──────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    # ── Read into DataFrame ──────────────────────────────────────
    try:
        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        log.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # ── Validate ─────────────────────────────────────────────────
    result = validate_dataframe(df)
    if not result["valid"]:
        raise HTTPException(
            status_code=422,
            detail={"message": "Validation failed", "errors": result["errors"]},
        )

    df = result["df"]
    log.info(f"  Validated: {len(df)} rows, {len(df.columns)} columns")

    # ── Save locally to data/raw/ ────────────────────────────────
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DATA_RAW_DIR / file.filename
    with open(local_path, "wb") as f_out:
        f_out.write(contents)
    log.info(f"  Saved locally: {local_path}")

    # ── Upload to Azure Blob (bronze container) ──────────────────
    blob_uploaded = upload_csv(str(local_path), f"uploads/{file.filename}")

    # ── Insert into Azure SQL ────────────────────────────────────
    init_tables()
    rows_inserted = insert_transactions(df)

    # ── Response ─────────────────────────────────────────────────
    return JSONResponse(
        status_code=200,
        content={
            "message":       "Ingestion successful",
            "filename":      file.filename,
            "rows_received": len(df),
            "rows_in_sql":   rows_inserted,
            "blob_uploaded": blob_uploaded,
            "warnings":      result["warnings"],
            "date_range": {
                "start": str(df["Date"].min().date()),
                "end":   str(df["Date"].max().date()),
            },
            "categories": df["Product Category"].value_counts().to_dict(),
        },
    )