# TruthLens — AI Fake News & Misinformation Detector

TruthLens is an academic college capstone research project designed to explore transparent, automated natural language processing (NLP) techniques for news classification and misinformation analysis. The application integrates a fine-tuned Transformer-based classification engine powered by FastAPI with a modern, responsive React dashboard.

---

## Table of Contents

- [Overview & Objectives](#overview--objectives)
- [Implemented Features](#implemented-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Local Installation & Setup (macOS)](#local-installation--setup-macos)
- [Running the Application](#running-the-application)
- [Automated Testing & Verification](#automated-testing--verification)
- [API Reference](#api-reference)
- [Model Architecture & Hardware Behavior](#model-architecture--hardware-behavior)
- [Responsible AI, Limitations & Known Risks](#responsible-ai-limitations--known-risks)
- [Project Status & Roadmap](#project-status--roadmap)
- [License](#license)

---

## Overview & Objectives

Digital misinformation and distorted headlines spread rapidly across social platforms and digital media. The objective of TruthLens is to construct an end-to-end, reproducible software prototype that:

1. **Ingests Text**: Accepts user-submitted news headlines, article excerpts, or claim statements.
2. **Evaluates Semantic Patterns**: Analyzes textual features through a fine-tuned BERT sequence-classification model.
3. **Outputs Calibrated Probabilities**: Returns normalized class probabilities (`FAKE` vs. `REAL`) along with model confidence scores.
4. **Emphasizes Responsible Presentation**: Contextualizes machine-learning predictions with prominent interpretation notices, clarifying that statistical model outputs do not constitute absolute factual verification.

---

## Implemented Features

- **Transformer Inference Pipeline**: Fine-tuned BERT sequence classification (`dima806/fake-news-classifier`) with 512-token truncation.
- **Hardware Acceleration**: Automatic device resolution supporting Apple Silicon GPU acceleration via Metal Performance Shaders (`mps`) with reliable fallback to `cpu`.
- **Lazy Loading & Singleton Caching**: Model weights and tokenizer load once on demand and remain cached in memory across requests.
- **Strict Input Validation**: Rejects empty strings, whitespace-only submissions, and payloads exceeding 50,000 characters with HTTP 422.
- **Sanitized Error Handling**: Gracefully handles model loading issues (HTTP 503) and inference failures (HTTP 500) without leaking internal stack traces or secrets.
- **Dual Endpoint Routing**: Supports both `POST /api/predict` and alias `POST /predict` for seamless local Vite development proxying and direct production routing.
- **Modern SaaS Dashboard**: Built with React 18, Vite, TypeScript, and Tailwind CSS.
- **Responsible Presentation Notice**: Prominently warns users about model uncertainty, dataset biases, and the distinction between statistical confidence and factual ground truth.
- **Demonstration Presets**: Includes pre-configured sample inputs for rapid interface testing, explicitly labeled as demonstration samples rather than verified facts.
- **Live Backend Health Indicator**: Real-time polling badge in navigation bar verifying connectivity against the `/health` endpoint.
- **Automated Test Coverage**: 24 backend pytest tests and 6 frontend unit tests validating schemas, error states, and regression isolation.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Layer (Browser)"]
        UI["React 18 + Vite Dashboard"]
        Input["Text Input (up to 50k chars)"]
        Verdict["Prediction Card & Probability Bars"]
    end

    subgraph DevProxy ["Development Proxy"]
        ViteProxy["Vite Dev Proxy (:5173/api)"]
    end

    subgraph Backend ["FastAPI Backend (:8000)"]
        API["FastAPI Application (app.main)"]
        Schema["Pydantic v2 Validation"]
        Service["FakeNewsClassifier Service"]
    end

    subgraph ML ["Inference Engine"]
        Tok["AutoTokenizer (BERT Cased)"]
        Model["BertForSequenceClassification"]
        Hardware["Apple Silicon MPS / CPU Fallback"]
        HFCache["Local HF Cache (~/.cache/huggingface)"]
    end

    UI -->|1. Submit text| Input
    Input -->|2. POST /api/predict| ViteProxy
    ViteProxy -->|3. Forward request| API
    API -->|4. Validate payload| Schema
    Schema -->|5. Predict| Service
    Service -->|6. Tokenize (max 512 tokens)| Tok
    Tok -->|7. Tensors| Model
    Model -->|8. Run forward pass| Hardware
    Hardware -->|9. Raw Logits| Service
    Service -->|10. Softmax & id2label mapping| API
    API -->|11. JSON Response| ViteProxy
    ViteProxy -->|12. PredictionResponse| UI
    UI -->|13. Render verdict & disclaimer| Verdict

    Service -.->|First load only| HFCache
```

---

## Technology Stack

### Backend
- **Python**: 3.11
- **Framework**: FastAPI (>=0.110.0) with Uvicorn
- **Validation**: Pydantic v2 (>=2.6.0)
- **Deep Learning**: PyTorch (>=2.2.0) with Apple Silicon MPS support
- **NLP / Model**: Hugging Face Transformers (>=4.38.0)
- **Testing**: Pytest (>=8.0.0), HTTPX (TestClient)

### Frontend
- **Framework**: React 18 with TypeScript
- **Bundler & Tooling**: Vite 5
- **Styling**: Tailwind CSS 3 with PostCSS and Autoprefixer
- **Icons**: Lucide React
- **Testing**: Node.js Native Test Runner (`node:test`)

---

## Repository Structure

```
fake-news-detector/
├── .gitignore                      # Git exclusion rules (virtual environments, caches, builds)
├── README.md                       # Comprehensive project documentation
├── backend/                        # FastAPI application & ML service
│   ├── .venv/                      # Python 3.11 virtual environment (gitignored)
│   ├── requirements.txt            # Python dependencies (FastAPI, PyTorch, Transformers)
│   ├── app/
│   │   ├── __init__.py             # App package definition
│   │   ├── config.py               # Model settings, input bounds, device configuration
│   │   ├── main.py                 # FastAPI application routes (/health, /, /api/predict)
│   │   ├── schemas/
│   │   │   ├── __init__.py         # Schemas package definition
│   │   │   └── prediction.py       # Pydantic request & response models
│   │   └── services/
│   │       ├── __init__.py         # Services package definition
│   │       └── classifier.py       # BERT classifier service, MPS/CPU fallback, singleton
│   └── tests/
│       ├── __init__.py             # Tests package definition
│       ├── test_classifier.py      # Unit tests for classifier service & regression isolation
│       ├── test_health.py          # Backend health check & root route tests
│       └── test_prediction.py      # Integration tests for /api/predict & error handling
└── frontend/                       # React + Vite + TypeScript application
    ├── index.html                  # HTML template & metadata
    ├── package.json                # Frontend dependencies & npm scripts
    ├── postcss.config.js           # PostCSS Tailwind plugins
    ├── tailwind.config.js          # Tailwind CSS palette & extensions
    ├── tsconfig.json               # TypeScript compiler configuration
    ├── vite.config.ts              # Vite server & /api proxy configuration
    ├── src/
    │   ├── main.tsx                # React DOM entrypoint
    │   ├── App.tsx                 # Dashboard view & state orchestration
    │   ├── index.css               # Global Tailwind directives & dark theme styling
    │   ├── vite-env.d.ts           # Vite client environment types
    │   ├── types/
    │   │   └── prediction.ts       # TypeScript API contract definitions
    │   ├── services/
    │   │   └── api.ts              # Typed API client, error parsing, and retry handling
    │   └── components/
    │       ├── Navbar.tsx          # Branding, model status, and live health indicator
    │       ├── Header.tsx          # Project overview & technology tags
    │       ├── InputSection.tsx    # Textarea, character counter, and demonstration presets
    │       ├── ResultsSection.tsx  # Prediction card, probability bars, and disclaimer
    │       ├── LoadingSection.tsx  # Accessible inference loading skeleton
    │       ├── ErrorSection.tsx    # Contextual error alerts (422, 503, 500, network)
    │       ├── EmptyState.tsx      # Awaiting analysis state with feature guide
    │       └── Footer.tsx          # System capabilities and documented limitations
    └── tests/
        └── api.test.mjs            # Frontend automated test suite (node:test)
```

---

## Prerequisites

Ensure your system meets the following prerequisites:
- **Operating System**: macOS (Apple Silicon recommended; Intel macOS and Linux supported with CPU inference)
- **Python**: 3.11 (`python3.11`)
- **Node.js**: v18.0.0 or higher (`node -v`)
- **Package Manager**: `npm` (v9 or higher)
- **Disk Space**: At least 1.5 GB free disk space (to store the ~433.5 MB model weights in the Hugging Face cache)
- **Internet Access**: Required on initial run to download model weights from Hugging Face Hub

---

## Local Installation & Setup (macOS)

### 1. Clone the Repository
```bash
git clone https://github.com/Jeel-dhameliya/fake-news-detector.git
cd fake-news-detector
```

### 2. Configure Python Virtual Environment (Backend)
If `backend/.venv` already exists, activate it. Otherwise, create it using Python 3.11:
```bash
# Create virtual environment if not present
python3.11 -m venv backend/.venv

# Activate virtual environment
source backend/.venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

To run the application locally, start the backend API and frontend development server in **two separate terminal windows**.

### Terminal 1: Backend Server (FastAPI)
From the repository root:
```bash
# Using direct path to virtual environment
./backend/.venv/bin/uvicorn app.main:app --app-dir backend --port 8000 --reload

# OR activating the virtual environment first
cd backend
source .venv/bin/activate
uvicorn app.main:app --port 8000 --reload
```
The FastAPI backend will start at: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

> **Note on First Run**: The first prediction request will trigger Hugging Face Transformers to download `dima806/fake-news-classifier` (~433.5 MB) to `~/.cache/huggingface/hub/`. Subsequent inferences use the local cache instantly.

### Terminal 2: Frontend Dashboard (Vite)
From the repository root:
```bash
cd frontend
npm run dev
```
The Vite development server will start at: `http://127.0.0.1:5173`
- The development server automatically proxies requests from `/api/*` to `http://127.0.0.1:8000/*`.
- Open `http://127.0.0.1:5173` in your browser.

---

## Automated Testing & Verification

All automated tests use mocked model dependencies by default, allowing test suites to run completely offline without downloading model weights.

### 1. Backend Test Suite (Pytest)
From the repository root:
```bash
./backend/.venv/bin/pytest backend/tests -v
```
**Test Coverage Includes**:
- Health check and root endpoint availability.
- Prediction request validation (rejecting empty strings, whitespace-only text, oversized text > 50,000 characters).
- Verified label mapping compliance (ensuring class index 0 and 1 are dynamically resolved).
- Probability sanity checks (ensuring non-negative, finite numbers summing to approximately 1.0).
- Hardware device resolution and graceful CPU fallback.
- Sanitized error handling for HTTP 503 and HTTP 500 without leaking stack traces.
- Regression test verifying distinct inputs are tokenized and processed independently without cross-input caching.

### 2. Frontend Test Suite (Node.js Native Test Runner)
From the `frontend/` directory:
```bash
cd frontend
npm test
```
**Test Coverage Includes**:
- Model prediction response schema parsing.
- HTTP 422 validation error message formatting.
- HTTP 503 model service unavailable handling.
- Network connection failure detection.
- Safe probability clamping and fallback handling (preventing `NaN%` displays).
- Demonstration sample verification against backend character limits.

### 3. Frontend Production Build Check
From the `frontend/` directory:
```bash
cd frontend
npm run build
```
Validates TypeScript compilation (`tsc -b`) and generates production assets in `frontend/dist/`.

---

## API Reference

### 1. Classify Statement
* **Route**: `POST /api/predict` (also available via alias `POST /predict`)
* **Content-Type**: `application/json`

#### Request Body
```json
{
  "text": "NASA confirmed today that the James Webb Space Telescope detected water vapor in the atmosphere of a rocky exoplanet orbiting within its host star habitable zone."
}
```

#### Response (200 OK)
```json
{
  "predicted_label": "REAL",
  "confidence": 0.9997,
  "model_confidence": 0.9997,
  "class_probabilities": {
    "FAKE": 0.0003,
    "REAL": 0.9997
  },
  "model_id": "dima806/fake-news-classifier",
  "device": "mps"
}
```

#### Validation Error (422 Unprocessable Entity)
Returned when input is empty, whitespace-only, missing, or exceeds 50,000 characters:
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "text"],
      "msg": "Value error, Input text cannot be empty or whitespace-only.",
      "input": "",
      "ctx": {"error": {}}
    }
  ]
}
```

#### Service Unavailable (503 Service Unavailable)
Returned if model weights fail to load:
```json
{
  "detail": "Classification model service is temporarily unavailable."
}
```

#### Server Error (500 Internal Server Error)
Returned on inference computation failure without leaking internal tracebacks:
```json
{
  "detail": "Inference processing failed. Please verify input text format and try again."
}
```

---

### 2. Health Check
* **Route**: `GET /health`
* **Response (200 OK)**:
```json
{
  "status": "healthy",
  "service": "TruthLens Backend",
  "version": "0.1.0"
}
```

### 3. Root Endpoint
* **Route**: `GET /`
* **Response (200 OK)**:
```json
{
  "message": "Welcome to TruthLens API - AI Fake News & Misinformation Detector",
  "docs_url": "/docs",
  "health_url": "/health",
  "version": "0.1.0"
}
```

---

## Model Architecture & Hardware Behavior

### Model Metadata
- **Hugging Face Model ID**: [`dima806/fake-news-classifier`](https://huggingface.co/dima806/fake-news-classifier)
- **Base Architecture**: `BertForSequenceClassification` (`google-bert/bert-base-cased`)
- **Classification Head**: Single-label binary classification (`problem_type: single_label_classification`)
- **Verified Label Mapping**:
  - `Index 0`: `FAKE`
  - `Index 1`: `REAL`
- **Tokenizer**: BertTokenizer (Cased, vocabulary size: 28,996 tokens)
- **Positional Limit**: Truncated to a maximum of 512 tokens
- **License**: Apache-2.0
- **Weight Footprint**:
  - `model.safetensors`: ~433.27 MB
  - `vocab.txt`: ~213 KB
  - `config.json`: ~779 B
  - Total download footprint: **~433.5 MB** stored in `~/.cache/huggingface/hub/`

### Hardware Acceleration & Fallback
The backend dynamically inspects available hardware devices at runtime:
1. **Apple Silicon (`mps`)**: If `torch.backends.mps.is_available()` and `is_built()` return true, the model is placed on Metal Performance Shaders for GPU acceleration.
2. **NVIDIA CUDA (`cuda`)**: If CUDA is detected, CUDA device placement is selected.
3. **CPU Fallback (`cpu`)**: If GPU acceleration is unavailable or if an unexpected forward-pass exception occurs on MPS, inference automatically retries on the CPU. The response contract strictly reports the actual device used (`mps` or `cpu`).

---

## Responsible AI, Limitations & Known Risks

TruthLens is an academic research prototype and must be assessed with appropriate context:

1. **Statistical Prediction vs. Ground Truth**:
   The output is a **machine-learning statistical classification**, not an assertion of factual truth. A `REAL` classification does not prove a claim is authentic, and a `FAKE` classification does not prove intent to deceive.
2. **Confidence Score Definition**:
   The `confidence` score represents the model's mathematical certainty over its fine-tuned class distribution. High confidence indicates that input phrasing matches statistical patterns observed in the training data, **not** that the claim is objectively factual.
3. **Benchmark Dataset Bias & Domain Shift**:
   The underlying model was fine-tuned on benchmark corpora (such as Kaggle/ISOT fake news datasets). These datasets often associate specific vocabulary (e.g., government agency names, political bylines, or short headlines) with particular classes. Short, out-of-distribution statements or contemporary events may produce counter-intuitive classifications.
4. **No Live Web Verification**:
   TruthLens does **not** crawl the live internet, verify external source URLs, cross-reference journalistic fact-checking registries, or evaluate domain reputation.
5. **No Token Heatmaps or XAI in Current Release**:
   Feature attribution algorithms (such as LIME or SHAP) and token-level saliency heatmaps are planned for future research milestones and are not implemented in the current release.

---

## Project Status & Roadmap

### Completed Milestones
- **Milestone 1**: Project scaffolding, FastAPI health endpoints, React + Vite skeleton, Tailwind CSS design system.
- **Milestone 2**: Hugging Face BERT classifier service, dynamic label mapping, MPS acceleration with CPU fallback, input length sanitization, and automated backend test suite.
- **Milestone 2.1**: Diagnostic audit identifying logit saturation and dataset domain sensitivity behaviors.
- **Milestone 3**: React dashboard integration, live health checks, demonstration samples, probability visualizer, and responsible interpretation notices.
- **Milestone 3.1 & Commit**: Pre-commit audit, whitespace hygiene, and commit `f5dd823` pushed to `main`.
- **Milestone 4A**: Comprehensive project documentation and evaluation guide.

### Planned Future Milestones
- **Source & URL Reputation Signals**: Domain classification based on audited journalistic trust datasets.
- **Explainable AI (XAI)**: Token-importance attribution (LIME/SHAP) highlighting words influencing model decisions.
- **Claim Extraction & Multi-Claim Parsing**: Deconstructing complex news articles into verifiable atomic claims.
- **Production Containerization**: Multi-stage Docker configuration with production reverse-proxy routing.

---

## License

Licensing terms for the TruthLens application codebase are currently pending institutional evaluation and final project submission review. The fine-tuned model weights (`dima806/fake-news-classifier`) are distributed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
