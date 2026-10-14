"""FastAPI application entrypoint for TruthLens."""

import logging
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.schemas.prediction import PredictionRequest, PredictionResponse
    from app.services.classifier import (
        FakeNewsClassifier,
        ModelInferenceError,
        ModelLoadError,
        get_classifier,
    )
except ImportError:
    from backend.app.schemas.prediction import PredictionRequest, PredictionResponse
    from backend.app.services.classifier import (
        FakeNewsClassifier,
        ModelInferenceError,
        ModelLoadError,
        get_classifier,
    )



logger = logging.getLogger(__name__)

app = FastAPI(
    title="TruthLens - AI Fake News & Misinformation Detector",
    description="Backend API for text classification, credibility assessment, and domain verification.",
    version="0.1.0",
)

# CORS configuration for frontend development
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def read_root():
    """Root endpoint welcoming clients and referencing API docs."""
    return {
        "message": "Welcome to TruthLens API - AI Fake News & Misinformation Detector",
        "docs_url": "/docs",
        "health_url": "/health",
        "version": "0.1.0",
    }


@app.get("/health", tags=["Health"])
def read_health():
    """Health check endpoint to verify backend operational readiness."""
    return {
        "status": "healthy",
        "service": "TruthLens Backend",
        "version": "0.1.0",
    }


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Classification"],
    summary="Classify news article text as FAKE or REAL",
)
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def predict_news(
    request: PredictionRequest,
    classifier: FakeNewsClassifier = Depends(get_classifier),
) -> PredictionResponse:
    """Classify input text using the fine-tuned BERT transformer pipeline.

    Accepts non-empty news text and returns predicted class label, confidence score,
    and class probability distribution.
    """
    try:
        result = classifier.predict(request.text)
        return PredictionResponse(**result)
    except ModelLoadError as exc:
        logger.error("Model service load failure during prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classification model service is temporarily unavailable.",
        ) from exc
    except ModelInferenceError as exc:
        logger.error("Model inference failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference processing failed. Please verify input text format and try again.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during classification prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the prediction.",
        ) from exc
