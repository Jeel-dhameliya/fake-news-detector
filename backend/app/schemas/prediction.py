"""Pydantic schemas for news classification prediction requests and responses."""

import unicodedata
from typing import Dict
from pydantic import BaseModel, Field, field_validator

try:
    from app.config import MAX_INPUT_LENGTH
except ImportError:
    from backend.app.config import MAX_INPUT_LENGTH



class PredictionRequest(BaseModel):
    """Request payload for news text classification."""

    text: str = Field(
        ...,
        description="The news article text, claim, or headline to classify.",
        examples=["NASA confirms detection of water vapor in rocky exoplanet atmosphere."],
    )

    @field_validator("text")
    @classmethod
    def validate_text_content(cls, value: str) -> str:
        """Validates that input text is non-empty, contains substantive visible content, and is within safe length limits."""
        if not isinstance(value, str):
            raise ValueError("Input text must be a string.")

        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Input text cannot be empty or whitespace-only.")

        # Reject inputs composed solely of zero-width, invisible formatting, or control characters
        has_substantive_content = any(
            not c.isspace() and unicodedata.category(c) not in ("Cf", "Cc", "Cs", "Cn")
            for c in value
        )
        if not has_substantive_content:
            raise ValueError("Input text cannot consist solely of invisible or zero-width characters.")

        if len(value) > MAX_INPUT_LENGTH:
            raise ValueError(
                f"Input text exceeds maximum allowed length of {MAX_INPUT_LENGTH} characters "
                f"(received {len(value)} characters)."
            )

        return value


class PredictionResponse(BaseModel):
    """Response payload for news classification prediction."""

    predicted_label: str = Field(
        ...,
        description="The predicted class label (e.g., 'FAKE' or 'REAL').",
        examples=["REAL"],
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the predicted label between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
        examples=[0.9852],
    )
    model_confidence: float = Field(
        ...,
        description="Explicitly labeled model confidence score (same as confidence).",
        ge=0.0,
        le=1.0,
        examples=[0.9852],
    )
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Normalized probability distribution across all verified model classes.",
        examples=[{"FAKE": 0.0148, "REAL": 0.9852}],
    )
    model_id: str = Field(
        ...,
        description="Model identifier used for inference.",
        examples=["dima806/fake-news-classifier"],
    )
    device: str = Field(
        ...,
        description="Hardware acceleration device used for inference (e.g., 'mps', 'cpu', 'cuda').",
        examples=["mps"],
    )
