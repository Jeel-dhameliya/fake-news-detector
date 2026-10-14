"""Services module for TruthLens backend."""

try:
    from backend.app.services.classifier import (
        FakeNewsClassifier,
        ModelInferenceError,
        ModelLoadError,
        get_classifier,
    )
except ImportError:
    from app.services.classifier import (
        FakeNewsClassifier,
        ModelInferenceError,
        ModelLoadError,
        get_classifier,
    )

__all__ = [
    "FakeNewsClassifier",
    "ModelLoadError",
    "ModelInferenceError",
    "get_classifier",
]
