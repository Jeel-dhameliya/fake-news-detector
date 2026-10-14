"""Schemas module for TruthLens API."""

try:
    from backend.app.schemas.prediction import PredictionRequest, PredictionResponse
except ImportError:
    from app.schemas.prediction import PredictionRequest, PredictionResponse

__all__ = ["PredictionRequest", "PredictionResponse"]
