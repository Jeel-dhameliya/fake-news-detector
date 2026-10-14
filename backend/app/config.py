"""Configuration settings for TruthLens backend application."""

import os
from typing import Literal

# Model settings
DEFAULT_MODEL_ID: str = os.getenv("TRUTHLENS_MODEL_ID", "dima806/fake-news-classifier")

# Text input limits
# 50,000 characters is approximately 7,000 to 10,000 words (safely covering full news articles)
MAX_INPUT_LENGTH: int = int(os.getenv("TRUTHLENS_MAX_INPUT_LENGTH", "50000"))
MIN_INPUT_LENGTH: int = int(os.getenv("TRUTHLENS_MIN_INPUT_LENGTH", "1"))

# Execution device: 'auto', 'mps', 'cuda', or 'cpu'
DEVICE_OVERRIDE: str = os.getenv("TRUTHLENS_DEVICE", "auto").lower()

# Verified default label mapping for dima806/fake-news-classifier
VERIFIED_ID2LABEL: dict[int, str] = {
    0: "FAKE",
    1: "REAL",
}
