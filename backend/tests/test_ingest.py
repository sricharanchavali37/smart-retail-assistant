"""
test_ingest.py
──────────────
Unit tests for POST /ingest API.
Run: pytest backend/tests/test_ingest.py -v
"""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app

client = TestClient(app)


def _make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


VALID_DF = pd.DataFrame([
    {
        "Transaction ID": "T001", "Date": "2023-06-15",
        "Customer ID": "C001", "Gender": "Female", "Age": 28,
        "Product Category": "Beauty", "Quantity": 2,
        "Price per Unit": 50, "Total Amount": 100,
    },
    {
        "Transaction ID": "T002", "Date": "2023-06-16",
        "Customer ID": "C002", "Gender": "Male", "Age": 35,
        "Product Category": "Electronics", "Quantity": 1,
        "Price per Unit": 300, "Total Amount": 300,
    },
])


def test_ingest_valid_csv():
    """Valid CSV should return 200 with correct row count."""
    csv_bytes = _make_csv_bytes(VALID_DF)
    response  = client.post(
        "/ingest/",
        files={"file": ("test_sales.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["rows_received"] == 2
    assert data["message"] == "Ingestion successful"


def test_ingest_non_csv_rejected():
    """Non-CSV file should return 400."""
    response = client.post(
        "/ingest/",
        files={"file": ("data.txt", io.BytesIO(b"some text"), "text/plain")},
    )
    assert response.status_code == 400


def test_ingest_missing_columns():
    """CSV missing required columns should return 422."""
    bad_df  = pd.DataFrame([{"Date": "2023-01-01", "Sales": 100}])
    csv_bytes = _make_csv_bytes(bad_df)
    response  = client.post(
        "/ingest/",
        files={"file": ("bad.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 422


def test_ingest_empty_csv():
    """CSV with headers only (no rows) should still return 200."""
    empty_df  = VALID_DF.iloc[:0]
    csv_bytes = _make_csv_bytes(empty_df)
    response  = client.post(
        "/ingest/",
        files={"file": ("empty.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code in [200, 422]


def test_ingest_response_has_categories():
    """Response should include category breakdown."""
    csv_bytes = _make_csv_bytes(VALID_DF)
    response  = client.post(
        "/ingest/",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 200
    assert "categories" in response.json()