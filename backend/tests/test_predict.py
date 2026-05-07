"""
test_predict.py
───────────────
Unit tests for GET /predict API.
Run: pytest backend/tests/test_predict.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app

client = TestClient(app)


def test_predict_beauty_7days():
    """Valid request for Beauty, 7 days ahead."""
    response = client.get("/predict/?category=Beauty&days_ahead=7")
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Beauty"
    assert len(data["forecast"]) == 7


def test_predict_electronics_14days():
    """Valid request for Electronics, 14 days ahead."""
    response = client.get("/predict/?category=Electronics&days_ahead=14")
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecast"]) == 14


def test_predict_clothing():
    """Valid request for Clothing."""
    response = client.get("/predict/?category=Clothing&days_ahead=3")
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Clothing"
    assert len(data["forecast"]) == 3


def test_predict_invalid_category():
    """Invalid category should return 400."""
    response = client.get("/predict/?category=Furniture&days_ahead=7")
    assert response.status_code == 400


def test_predict_days_out_of_range():
    """days_ahead > 14 should return 422."""
    response = client.get("/predict/?category=Beauty&days_ahead=99")
    assert response.status_code == 422


def test_predict_response_structure():
    """Response must contain forecast list and summary dict."""
    response = client.get("/predict/?category=Clothing&days_ahead=5")
    assert response.status_code == 200
    data = response.json()
    assert "forecast"  in data
    assert "summary"   in data
    assert "total_predicted_revenue" in data["summary"]
    assert "anomaly_days_count"      in data["summary"]


def test_predict_revenue_non_negative():
    """All predicted revenues must be >= 0."""
    response = client.get("/predict/?category=Beauty&days_ahead=7")
    assert response.status_code == 200
    for day in response.json()["forecast"]:
        assert day["predicted_revenue"] >= 0