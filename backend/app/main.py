from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/")
def read_root():
    return {
        "message": "Welcome to TruthLens API - AI Fake News & Misinformation Detector",
        "docs_url": "/docs",
        "health_url": "/health",
        "version": "0.1.0",
    }


@app.get("/health")
def read_health():
    return {
        "status": "healthy",
        "service": "TruthLens Backend",
        "version": "0.1.0",
    }
