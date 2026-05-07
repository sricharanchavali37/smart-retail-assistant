"""
test_agent.py
─────────────
Unit tests for POST /agent API.
Run: pytest backend/tests/test_agent.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.main import app

client = TestClient(app)


def test_agent_forecast_routing():
    """Forecast-related query should route to forecast_insight agent."""
    response = client.post("/agent/", json={"message": "predict demand for next week"})
    assert response.status_code == 200
    data = response.json()
    assert data["agent"] == "forecast_insight"


def test_agent_analyst_routing():
    """Anomaly-related query should route to retail_analyst agent."""
    response = client.post("/agent/", json={"message": "show me anomalies in sales"})
    assert response.status_code == 200
    data = response.json()
    assert data["agent"] == "retail_analyst"


def test_agent_knowledge_routing():
    """Policy-related query should route to product_knowledge agent."""
    response = client.post("/agent/", json={"message": "what is the return policy?"})
    assert response.status_code == 200
    data = response.json()
    assert data["agent"] == "product_knowledge"


def test_agent_response_has_required_fields():
    """Response must contain agent, response, and status fields."""
    response = client.post("/agent/", json={"message": "hello, what can you do?"})
    assert response.status_code == 200
    data = response.json()
    assert "agent"    in data
    assert "response" in data
    assert "status"   in data


def test_agent_with_session_id():
    """Session ID should be echoed back in response."""
    response = client.post(
        "/agent/",
        json={"message": "which product sells most?", "session_id": "test-session-42"},
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "test-session-42"


def test_agent_empty_message_rejected():
    """Empty message should return 422."""
    response = client.post("/agent/", json={"message": ""})
    assert response.status_code == 422