import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Ensure backend root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fastapi.testclient import TestClient
from app.main import app
from app.services.classifier import FakeNewsClassifier

client = TestClient(app)


def test_read_root():
    """Verify root welcome route returns expected documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "TruthLens API" in data["message"]
    assert data["version"] == "0.1.0"
    assert data["health_url"] == "/health"
    assert data["api_health_url"] == "/api/health"


def test_health_check():
    """Verify standard /health endpoint returns backward-compatible payload and liveness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "TruthLens Backend"
    assert data["version"] == "0.1.0"
    assert data["api_liveness"] == "alive"
    assert "model_readiness" in data
    assert "model_loaded" in data


def test_health_route_parity():
    """Verify GET /api/health and GET /health return identical contracts and status 200."""
    res_root = client.get("/health")
    res_api = client.get("/api/health")

    assert res_root.status_code == 200
    assert res_api.status_code == 200
    assert res_root.json() == res_api.json()


def test_health_check_does_not_trigger_model_load():
    """Verify calling health endpoints does not initialize or load model weights."""
    FakeNewsClassifier.reset_instance()
    assert FakeNewsClassifier._instance is None

    # Call /health and /api/health
    res_root = client.get("/health")
    res_api = client.get("/api/health")

    assert res_root.status_code == 200
    assert res_api.status_code == 200

    data = res_root.json()
    assert data["api_liveness"] == "alive"
    assert data["model_readiness"] == "unloaded"
    assert data["model_loaded"] is False
    assert data["device"] is None

    # Confirm model singleton was not instantiated or loaded
    assert FakeNewsClassifier._instance is None


def test_health_accurately_reflects_loaded_model_state():
    """Verify health endpoints accurately reflect model readiness when initialized without reloading."""
    try:
        # Create a mock classifier singleton with loaded state
        mock_instance = MagicMock()
        mock_instance._is_loaded = True
        mock_instance.device = torch.device("cpu")
        mock_instance.model_id = "dima806/fake-news-classifier"

        FakeNewsClassifier._instance = mock_instance

        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["api_liveness"] == "alive"
        assert data["model_readiness"] == "ready"
        assert data["model_loaded"] is True
        assert data["device"] == "cpu"
        assert data["model_id"] == "dima806/fake-news-classifier"
    finally:
        FakeNewsClassifier.reset_instance()
