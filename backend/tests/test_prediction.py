"""Integration and schema tests for POST /api/predict endpoint.

Ensures valid schemas, validation on empty/oversized inputs, mocked model output
integration, and proper HTTP error statuses without model weight downloads.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.services.classifier import (
    FakeNewsClassifier,
    ModelInferenceError,
    ModelLoadError,
    get_classifier,
)
from app.config import MAX_INPUT_LENGTH

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_dependency_overrides():
    """Ensure FastAPI dependency overrides are cleared after each test."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def test_predict_valid_request_success():
    """Verify standard valid request returns 200 and matches expected schema."""
    mock_classifier = MagicMock(spec=FakeNewsClassifier)
    mock_classifier.predict.return_value = {
        "predicted_label": "REAL",
        "confidence": 0.9852,
        "model_confidence": 0.9852,
        "class_probabilities": {"FAKE": 0.0148, "REAL": 0.9852},
        "model_id": "dima806/fake-news-classifier",
        "device": "cpu",
    }
    app.dependency_overrides[get_classifier] = lambda: mock_classifier

    payload = {"text": "Government space agency discovers water vapor on exoplanet."}
    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] == "REAL"
    assert data["confidence"] == 0.9852
    assert data["model_confidence"] == 0.9852
    assert data["class_probabilities"] == {"FAKE": 0.0148, "REAL": 0.9852}
    assert data["model_id"] == "dima806/fake-news-classifier"
    assert data["device"] == "cpu"
    mock_classifier.predict.assert_called_once_with(payload["text"])


def test_predict_alias_endpoint_works():
    """Verify /predict alias route works identically for Vite proxy support."""
    mock_classifier = MagicMock(spec=FakeNewsClassifier)
    mock_classifier.predict.return_value = {
        "predicted_label": "FAKE",
        "confidence": 0.9412,
        "model_confidence": 0.9412,
        "class_probabilities": {"FAKE": 0.9412, "REAL": 0.0588},
        "model_id": "dima806/fake-news-classifier",
        "device": "mps",
    }
    app.dependency_overrides[get_classifier] = lambda: mock_classifier

    response = client.post("/predict", json={"text": "Miracle cure revealed by secret whistleblower!"})
    assert response.status_code == 200
    assert response.json()["predicted_label"] == "FAKE"


def test_predict_empty_text_returns_422():
    """Verify empty text string is rejected with HTTP 422 Unprocessable Entity."""
    response = client.post("/api/predict", json={"text": ""})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    # Ensure error mentions empty or whitespace-only
    assert any("empty or whitespace" in str(err) for err in data["detail"])


def test_predict_whitespace_only_returns_422():
    """Verify whitespace-only text is rejected with HTTP 422."""
    response = client.post("/api/predict", json={"text": "   \n\t  \r  "})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any("empty or whitespace" in str(err) for err in data["detail"])


def test_predict_oversized_text_returns_422():
    """Verify text exceeding MAX_INPUT_LENGTH is safely rejected with HTTP 422."""
    oversized_text = "A" * (MAX_INPUT_LENGTH + 10)
    response = client.post("/api/predict", json={"text": oversized_text})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any("exceeds maximum allowed length" in str(err) for err in data["detail"])


def test_predict_missing_text_field_returns_422():
    """Verify request missing the text field is rejected with HTTP 422."""
    response = client.post("/api/predict", json={"title": "Missing text"})
    assert response.status_code == 422


def test_predict_invalid_json_returns_422():
    """Verify invalid JSON payload is rejected."""
    response = client.post(
        "/api/predict",
        content="not a valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_model_load_failure_returns_503():
    """Verify ModelLoadError returns HTTP 503 without leaking stack traces."""
    mock_classifier = MagicMock(spec=FakeNewsClassifier)
    mock_classifier.predict.side_effect = ModelLoadError(
        "Classification model 'dima806/fake-news-classifier' failed to load."
    )
    app.dependency_overrides[get_classifier] = lambda: mock_classifier

    response = client.post("/api/predict", json={"text": "Valid headline text for test."})
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert "temporarily unavailable" in data["detail"].lower()
    # Confirm internal stack trace is not exposed
    assert "Traceback" not in response.text


def test_predict_model_inference_failure_returns_500():
    """Verify ModelInferenceError returns HTTP 500 without leaking stack traces."""
    mock_classifier = MagicMock(spec=FakeNewsClassifier)
    mock_classifier.predict.side_effect = ModelInferenceError(
        "Model inference computation failed."
    )
    app.dependency_overrides[get_classifier] = lambda: mock_classifier

    response = client.post("/api/predict", json={"text": "Valid headline text for test."})
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "failed" in data["detail"].lower()
    assert "Traceback" not in response.text


def test_health_check_remains_functional():
    """Verify health endpoint remains healthy after adding prediction routes."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "TruthLens Backend"
    assert data["version"] == "0.1.0"
    assert data["api_liveness"] == "alive"


def test_root_endpoint_remains_functional():
    """Verify root endpoint remains functional."""
    response = client.get("/")
    assert response.status_code == 200
    assert "TruthLens API" in response.json()["message"]


def test_predict_zero_width_space_returns_422():
    """Verify text consisting solely of zero-width spaces (U+200B) is rejected with HTTP 422."""
    for payload_text in ["\u200b", "   \u200b\u200b   ", "\u200b\u200b\u200b"]:
        response = client.post("/api/predict", json={"text": payload_text})
        assert response.status_code == 422
        data = response.json()
        assert any("invisible or zero-width" in str(err) for err in data["detail"])


def test_predict_bom_zwnbs_returns_422():
    """Verify text consisting solely of BOM/zero-width no-break spaces (U+FEFF) is rejected with HTTP 422."""
    for payload_text in ["\ufeff", " \ufeff ", "\ufeff\u200b"]:
        response = client.post("/api/predict", json={"text": payload_text})
        assert response.status_code == 422
        data = response.json()
        assert any("invisible or zero-width" in str(err) for err in data["detail"])


def test_predict_embedded_invisible_character_in_valid_text_succeeds():
    """Verify legitimate text containing embedded formatting characters is accepted as-is without broad mutation."""
    mock_classifier = MagicMock(spec=FakeNewsClassifier)
    mock_classifier.predict.return_value = {
        "predicted_label": "REAL",
        "confidence": 0.95,
        "model_confidence": 0.95,
        "class_probabilities": {"FAKE": 0.05, "REAL": 0.95},
        "model_id": "dima806/fake-news-classifier",
        "device": "cpu",
    }
    app.dependency_overrides[get_classifier] = lambda: mock_classifier

    # Valid sentence with an embedded zero-width space
    valid_text_with_zwsp = "Government\u200b announcement regarding regional public health initiatives."
    response = client.post("/api/predict", json={"text": valid_text_with_zwsp})

    assert response.status_code == 200
    assert response.json()["predicted_label"] == "REAL"
    # Ensure text was passed as supplied to classifier without destructive mutation
    mock_classifier.predict.assert_called_once_with(valid_text_with_zwsp)


def test_predict_malformed_json_returns_422_on_both_routes():
    """Verify malformed JSON payloads return controlled HTTP 422 error on both prediction routes."""
    for route in ["/api/predict", "/predict"]:
        response = client.post(
            route,
            content="{invalid json format: missing quotes",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "detail" in response.json()


def test_predict_json_array_payload_returns_422():
    """Verify JSON array payloads return controlled HTTP 422 on both routes."""
    for route in ["/api/predict", "/predict"]:
        response = client.post(route, json=[{"text": "valid text inside array"}])
        assert response.status_code == 422
        assert "detail" in response.json()


def test_predict_scalar_json_payload_returns_422():
    """Verify scalar JSON payloads return controlled HTTP 422."""
    for scalar in ["plain scalar string", 12345, True]:
        response = client.post("/api/predict", json=scalar)
        assert response.status_code == 422
        assert "detail" in response.json()


def test_predict_unsupported_content_type_returns_422():
    """Verify unsupported content types like text/plain return controlled HTTP 422."""
    for route in ["/api/predict", "/predict"]:
        response = client.post(
            route,
            content="Headline statement submitted as raw plain text",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 422
        assert "detail" in response.json()
