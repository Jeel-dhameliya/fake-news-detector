"""Fake news classification service using Hugging Face Transformers and PyTorch.

Supports lazy model loading, cached singleton instances, Apple Silicon MPS
acceleration with CPU fallback, and verified class-label mapping.
"""

import logging
import math
import threading
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from app.config import DEFAULT_MODEL_ID, DEVICE_OVERRIDE, VERIFIED_ID2LABEL
except ImportError:
    from backend.app.config import DEFAULT_MODEL_ID, DEVICE_OVERRIDE, VERIFIED_ID2LABEL


logger = logging.getLogger(__name__)



class ModelLoadError(Exception):
    """Raised when the classification model or tokenizer cannot be loaded."""


class ModelInferenceError(Exception):
    """Raised when model inference execution fails."""


class FakeNewsClassifier:
    """Service encapsulating the Transformer fake-news classification pipeline."""

    _instance: Optional["FakeNewsClassifier"] = None
    _lock = threading.Lock()

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = self._resolve_device(device or DEVICE_OVERRIDE)
        self.tokenizer: Any = None
        self.model: Any = None
        self.id2label: Dict[int, str] = VERIFIED_ID2LABEL.copy()
        self._is_loaded = False
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_id: str = DEFAULT_MODEL_ID) -> "FakeNewsClassifier":
        """Get or initialize singleton instance of the classifier."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_id=model_id)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing)."""
        with cls._lock:
            cls._instance = None

    def _resolve_device(self, preferred: str) -> torch.device:
        """Resolve the optimal execution device with Apple Silicon MPS support."""
        pref = preferred.lower()
        if pref == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            logger.warning("MPS device requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        elif pref == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA device requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        elif pref == "cpu":
            return torch.device("cpu")

        # Auto detection: MPS (Apple Silicon) -> CUDA -> CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load(self) -> None:
        """Lazily load tokenizer and model weights and establish verified label mapping."""
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:
                return

            try:
                logger.info("Loading tokenizer for model %s", self.model_id)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

                logger.info("Loading model weights for %s", self.model_id)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
                self.model.eval()

                # Attempt device assignment with graceful CPU fallback
                try:
                    self.model.to(self.device)
                except Exception as dev_err:
                    logger.warning(
                        "Failed to place model on device %s (%s). Falling back to CPU.",
                        self.device,
                        dev_err,
                    )
                    self.device = torch.device("cpu")
                    self.model.to(self.device)

                # Extract and verify label mapping from model config
                self._extract_label_mapping()
                self._is_loaded = True
                logger.info(
                    "Model %s successfully loaded on %s with labels %s",
                    self.model_id,
                    self.device,
                    self.id2label,
                )
            except Exception as exc:
                logger.error("Failed to load model %s: %s", self.model_id, exc)
                raise ModelLoadError(
                    f"Classification model '{self.model_id}' failed to load."
                ) from exc

    def _extract_label_mapping(self) -> None:
        """Extract id2label from model config, ensuring verified mappings."""
        raw_mapping = getattr(self.model.config, "id2label", None)
        if raw_mapping and isinstance(raw_mapping, dict):
            parsed = {}
            for k, v in raw_mapping.items():
                try:
                    parsed[int(k)] = str(v).upper()
                except (ValueError, TypeError):
                    continue

            # Check if extracted mapping contains standard label placeholders
            if parsed and not all(val.startswith("LABEL_") for val in parsed.values()):
                self.id2label = parsed
                return

        # Fallback to verified configuration mapping
        self.id2label = VERIFIED_ID2LABEL.copy()

    def predict(self, text: str) -> Dict[str, Any]:
        """Perform classification on input text and return structured predictions."""
        if not self._is_loaded:
            self.load()

        # Tokenize text with 512-token max length truncation
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as exc:
            logger.error("Tokenization error: %s", exc)
            raise ModelInferenceError("Input tokenization failed.") from exc

        # Model forward pass with MPS to CPU fallback
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
        except Exception as exc:
            if self.device.type == "mps":
                logger.warning(
                    "MPS forward pass failed (%s). Retrying inference on CPU.", exc
                )
                try:
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    cpu_inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**cpu_inputs)
                        logits = outputs.logits
                except Exception as cpu_exc:
                    logger.error("Inference retry on CPU failed: %s", cpu_exc)
                    raise ModelInferenceError("Model inference computation failed.") from cpu_exc
            else:
                logger.error("Inference execution failed: %s", exc)
                raise ModelInferenceError("Model inference computation failed.") from exc

        # Calculate softmax probabilities
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        # Handle edge case where single float is returned or invalid probabilities
        if isinstance(probs, float):
            probs = [probs]

        if not all(math.isfinite(p) for p in probs):
            logger.error("Non-finite probability values detected in model output: %s", probs)
            raise ModelInferenceError("Model returned invalid non-finite probability scores.")

        # Map probabilities using verified id2label
        class_probabilities: Dict[str, float] = {}
        for idx, prob in enumerate(probs):
            label = self.id2label.get(idx, f"CLASS_{idx}")
            class_probabilities[label] = round(float(prob), 4)

        # Determine predicted label via argmax
        predicted_idx = int(torch.argmax(logits, dim=-1).item())
        predicted_label = self.id2label.get(predicted_idx, f"CLASS_{predicted_idx}")
        confidence = class_probabilities.get(predicted_label, 0.0)

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "model_confidence": confidence,
            "class_probabilities": class_probabilities,
            "model_id": self.model_id,
            "device": str(self.device.type),
        }


def get_classifier() -> FakeNewsClassifier:
    """Dependency provider returning singleton instance of FakeNewsClassifier."""
    return FakeNewsClassifier.get_instance()
