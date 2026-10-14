"""Unit tests for FakeNewsClassifier service with mocked Transformer dependencies.

Ensures lazy loading, device resolution, label mappings, probability calculations,
and error handling are thoroughly tested without requiring real model downloads.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

# Ensure backend root is on sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.classifier import (
    FakeNewsClassifier,
    ModelInferenceError,
    ModelLoadError,
)



@pytest.fixture(autouse=True)
def reset_classifier_singleton():
    """Ensure each test starts with a clean classifier state."""
    FakeNewsClassifier.reset_instance()
    yield
    FakeNewsClassifier.reset_instance()


class MockModelOutput:
    """Mock container mimicking Transformers SequenceClassifierOutput."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


def test_device_resolution_auto_apple_silicon():
    """Verify device resolution correctly selects MPS when available."""
    with patch("torch.backends.mps.is_available", return_value=True), patch(
        "torch.backends.mps.is_built", return_value=True
    ):
        classifier = FakeNewsClassifier(device="auto")
        assert classifier.device.type == "mps"


def test_device_resolution_fallback_to_cpu():
    """Verify device resolution falls back to CPU when MPS and CUDA are absent."""
    with patch("torch.backends.mps.is_available", return_value=False), patch(
        "torch.cuda.is_available", return_value=False
    ):
        classifier = FakeNewsClassifier(device="auto")
        assert classifier.device.type == "cpu"


def test_device_resolution_explicit_cpu():
    """Verify device resolution respects explicit CPU preference."""
    classifier = FakeNewsClassifier(device="cpu")
    assert classifier.device.type == "cpu"


def test_singleton_instance_caching():
    """Verify FakeNewsClassifier maintains a singleton instance."""
    inst1 = FakeNewsClassifier.get_instance()
    inst2 = FakeNewsClassifier.get_instance()
    assert inst1 is inst2


def test_lazy_loading_not_loaded_until_needed():
    """Verify model is not loaded upon construction."""
    classifier = FakeNewsClassifier(model_id="mock-model")
    assert classifier._is_loaded is False
    assert classifier.model is None
    assert classifier.tokenizer is None


def test_mocked_model_loading_success():
    """Verify load() loads tokenizer, model, and verifies label mapping."""
    classifier = FakeNewsClassifier(model_id="mock/fake-news")

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.config.id2label = {"0": "FAKE", "1": "REAL"}

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
    ) as mock_tok_loader, patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model,
    ) as mock_mod_loader:
        classifier.load()

        assert classifier._is_loaded is True
        assert classifier.id2label == {0: "FAKE", 1: "REAL"}
        mock_tok_loader.assert_called_once_with("mock/fake-news")
        mock_mod_loader.assert_called_once_with("mock/fake-news")
        mock_model.eval.assert_called_once()


def test_mocked_model_loading_failure_raises_model_load_error():
    """Verify load() raises ModelLoadError on network or I/O failure."""
    classifier = FakeNewsClassifier(model_id="nonexistent-model")

    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=OSError("Repository not found or weights unavailable"),
    ):
        with pytest.raises(ModelLoadError) as exc_info:
            classifier.load()
        assert "failed to load" in str(exc_info.value)


def test_label_mapping_never_assumes_fixed_index():
    """Crucial requirement: Verify predictions honor model config id2label even if inverted."""
    classifier = FakeNewsClassifier(model_id="inverted-labels-model", device="cpu")

    # Inverted label mapping: index 0 is REAL, index 1 is FAKE
    classifier.id2label = {0: "REAL", 1: "FAKE"}
    classifier._is_loaded = True

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    classifier.tokenizer = mock_tokenizer

    # Model outputs higher logit for index 0 (which is REAL under this mapping)
    mock_model = MagicMock()
    mock_model.return_value = MockModelOutput(
        logits=torch.tensor([[3.5, -1.2]])
    )
    classifier.model = mock_model

    result = classifier.predict("Sample breaking news")
    assert result["predicted_label"] == "REAL"
    assert result["class_probabilities"]["REAL"] > result["class_probabilities"]["FAKE"]
    assert result["confidence"] == result["class_probabilities"]["REAL"]
    assert result["model_confidence"] == result["confidence"]


def test_predict_prob_sum_and_finite_values():
    """Verify prediction outputs have finite probabilities summing to ~1.0."""
    classifier = FakeNewsClassifier(model_id="mock-model", device="cpu")
    classifier.id2label = {0: "FAKE", 1: "REAL"}
    classifier._is_loaded = True

    classifier.tokenizer = MagicMock(
        return_value={
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
    )
    classifier.model = MagicMock(
        return_value=MockModelOutput(logits=torch.tensor([[-2.0, 4.0]]))
    )

    result = classifier.predict("Confirmed scientific discovery announced.")
    assert result["predicted_label"] == "REAL"
    probs = result["class_probabilities"]
    assert "FAKE" in probs and "REAL" in probs
    assert 0.0 <= probs["FAKE"] <= 1.0
    assert 0.0 <= probs["REAL"] <= 1.0
    assert pytest.approx(probs["FAKE"] + probs["REAL"], abs=0.01) == 1.0


def test_inference_failure_raises_model_inference_error():
    """Verify inference failure raises ModelInferenceError cleanly."""
    classifier = FakeNewsClassifier(model_id="mock-model", device="cpu")
    classifier._is_loaded = True
    classifier.tokenizer = MagicMock(
        return_value={
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
    )
    classifier.model = MagicMock(side_effect=RuntimeError("CUDA/MPS out of memory"))

    with pytest.raises(ModelInferenceError) as exc_info:
        classifier.predict("Some news text")
    assert "failed" in str(exc_info.value).lower()


def test_inference_pipeline_processes_different_inputs_separately():
    """Regression test: Verify different inputs are tokenized and processed independently without cross-input caching or tensor leakage."""
    classifier = FakeNewsClassifier(model_id="mock-model", device="cpu")
    classifier.id2label = {0: "FAKE", 1: "REAL"}
    classifier._is_loaded = True

    captured_texts = []

    def mock_tokenize(text, **kwargs):
        captured_texts.append(text)
        length = len(text)
        return {
            "input_ids": torch.tensor([[101, length, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    classifier.tokenizer = MagicMock(side_effect=mock_tokenize)

    def mock_forward(**inputs):
        token_val = float(inputs["input_ids"][0, 1].item())
        # Scale to realistic logit range [-3.0, 3.0]
        return MockModelOutput(logits=torch.tensor([[token_val * 0.05, 1.0 - token_val * 0.05]]))


    classifier.model = MagicMock(side_effect=mock_forward)

    res_a = classifier.predict("Input text alpha")
    res_b = classifier.predict("Input text beta with different length")
    res_a_repeat = classifier.predict("Input text alpha")

    assert captured_texts == [
        "Input text alpha",
        "Input text beta with different length",
        "Input text alpha",
    ]
    assert res_a["class_probabilities"] != res_b["class_probabilities"]
    assert res_a["class_probabilities"] == res_a_repeat["class_probabilities"]
    assert res_a["confidence"] == res_a_repeat["confidence"]
