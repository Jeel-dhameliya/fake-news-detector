# TruthLens: AI Fake News & Misinformation Detector
## College Project Evaluation — 5-Minute Timed Speaker Script

**Project Repository:** `/Users/jeel/Desktop/fake-news-detector`  
**Latest Verified Commit:** `c99f6f7adbc2e466aaf3d5da5577cc29f27c7c25`  
**Target Duration:** 5 Minutes (300 seconds)  
**Format:** Oral Defense & Live Demonstration Walkthrough  

---

### Slide 1: Title & Technology Summary
- **Slide Title:** TruthLens: AI Fake News & Misinformation Detector
- **Subtitle:** Academic Capstone Project Evaluation (2026)
- **Allocated Time:** 0:00 – 0:30 (30 seconds)

#### Speaker Script:
> "Good morning, respected professors and evaluation committee members. Today, I am proud to present **TruthLens**, an academic capstone research prototype designed for automated fake news and misinformation text classification.
>
> TruthLens combines a modern, responsive React 18 and TypeScript frontend with a high-performance Python FastAPI backend, executing transformer inference using PyTorch and Hugging Face with Apple Silicon GPU hardware acceleration.
>
> Over the course of this project, our primary focus was not merely assembling an inference pipeline, but engineering a resilient, transparent system that explicitly confronts and documents the real-world limitations of automated language models when applied to factual claim classification. Let us begin with the problem context."

---

### Slide 2: Problem Context & Motivation
- **Slide Title:** Problem Context & Motivation
- **Subtitle:** The Challenge of Misinformation in the Digital Era
- **Allocated Time:** 0:30 – 1:00 (30 seconds)

#### Speaker Script:
> "The exponential speed of social media distribution makes digital misinformation an urgent challenge for public health, elections, and civic cohesion. While human fact-checking organizations like PolitiFact and Snopes provide gold-standard verification, they cannot scale to monitor millions of claims per hour.
>
> Naturally, the machine learning community turned toward natural language processing and transformer models to automate claim detection. However, there is a fundamental scientific misconception that must be dispelled: a neural network classification head produces a statistical probability distribution over learned vocabulary patterns—it does *not* possess an internal world model or a live factual database.
>
> If an AI classifier reports a 99% score, that represents linguistic pattern matching over training labels, not verified factual truth. TruthLens was built specifically to study this distinction and provide transparent disclosures to users."

---

### Slide 3: Project Goals & Scope
- **Slide Title:** Project Goals & Scope
- **Subtitle:** Core Architectural & Research Objectives
- **Allocated Time:** 1:00 – 1:30 (30 seconds)

#### Speaker Script:
> "To address this challenge rigorously, we established four core engineering objectives:
>
> First, implement an end-to-end text classification service capable of tokenizing input statements and performing transformer inference with Apple Silicon MPS GPU acceleration and dynamic CPU fallback.
>
> Second, build an intuitive, accessible web dashboard featuring responsive design, dynamic character counting, pre-populated test samples, and prominent confidence metrics.
>
> Third, construct a robust API integration backed by dual health-monitoring endpoints—`GET /health` and `GET /api/health`—which report API liveness and model service state without prematurely loading weights, alongside 15-second client timeouts and automated network recovery.
>
> And fourth, maintain strict academic honesty by displaying explicit UI disclosures and benchmarking our model against an authoritative external test dataset to quantify its true out-of-domain capabilities."

---

### Slide 4: Engineering Design & System Architecture
- **Slide Title:** Engineering Design
- **Subtitle:** Decoupled Client-Server System Architecture
- **Allocated Time:** 1:30 – 2:05 (35 seconds)

#### Speaker Script:
> "Let us examine the architecture of TruthLens, which follows a decoupled, three-tier client-server topology.
>
> At the presentation layer, our frontend is built with React 18, TypeScript, and Tailwind CSS. It features a custom polling hook that checks backend health every 30 seconds, automatically surfacing network state transitions.
>
> At the application layer, our backend runs on FastAPI with Uvicorn. We enforce strict input validation using Pydantic, which enforces a 1 to 50,000 character length limit and rejects submissions that are whitespace-only or consist solely of invisible Unicode or zero-width formatting characters.
>
> At the machine learning layer, we load a pre-trained sequence classification model, `dima806/fake-news-classifier`, based on the 12-layer BERT architecture. Inference is executed in PyTorch with lazy singleton initialization and Apple Silicon MPS GPU hardware acceleration with dynamic CPU fallback. Both tiers operate under strict 15-second request timeouts to guarantee system stability."

---

### Slide 5: Inference Pipeline & Mechanics
- **Slide Title:** Inference Pipeline & Mechanics
- **Subtitle:** Step-by-Step Execution from Raw Input to Softmax Score
- **Allocated Time:** 2:05 – 2:40 (35 seconds)

#### Speaker Script:
> "When a user submits a claim into TruthLens, the system executes a four-stage pipeline:
>
> First, input validation ensures the text contains between 1 and 50,000 characters and rejects payloads that are empty, whitespace-only, or composed entirely of invisible or zero-width formatting characters.
>
> Second, BERT’s WordPiece tokenizer breaks the claim into subword tokens, prepends the classification token `[CLS]`, appends `[SEP]`, and produces input IDs and attention masks bounded to a 512-token sequence.
>
> Third, the tensors pass through 12 transformer encoder layers on our hardware device, and the classification head outputs two unnormalized raw logits.
>
> Fourth, a standard softmax function normalizes these logits into output probabilities between 0 and 1.
>
> Here we reinforce our central technical message: softmax scores measure the model’s internal class preference given its training weights. They are *not* calibrated probabilities of objective real-world facts."

---

### Slide 6: Live Application Demonstration
- **Slide Title:** Live Application Demonstration
- **Subtitle:** Demonstrating Model Behavior Across Diverse Sample Types
- **Allocated Time:** 2:40 – 3:20 (40 seconds)

#### Speaker Script:
> "Now, let us examine our live test demonstrations, which highlight both the strengths and the acute vulnerabilities of text-only classification.
>
> In our first sample, we tested a structured scientific statement asserting that NASA’s James Webb Space Telescope detected exoplanet water vapor. In our recorded evaluation run, TruthLens classified this as REAL with a 100.0% model output score.
>
> However, observe our second sample: an unfounded conspiracy headline asserting that the government is secretly installing mind-control microchips into tap water. In the recorded run, the classifier also classified this statement as REAL with a 100.0% output score!
>
> This demonstrates a clear failure case for the current classifier. Because the model operates purely on statistical text patterns without external evidence retrieval or fact-checking knowledge bases, it cannot evaluate factual veracity. High model confidence on an absurd claim shows that softmax output scores cannot be treated as evidence of real-world truth.
>
> This live finding directly justifies why TruthLens displays prominent UI notices clarifying that 'Model Output Score' represents statistical label preference rather than factual truth."

---

### Slide 7: Rigorous Out-of-Domain Evaluation
- **Slide Title:** Rigorous Out-of-Domain Evaluation
- **Subtitle:** LIAR Benchmark Methodology & Confusion Matrix Analysis
- **Allocated Time:** 3:20 – 4:00 (40 seconds)

#### Speaker Script:
> "To establish the true predictive capability of the model beyond isolated examples, we subjected TruthLens to an out-of-domain evaluation using the prestigious LIAR benchmark from William Yang Wang, published at ACL 2017.
>
> Using the official `test.tsv` split verified by SHA-256 checksums, we excluded 267 ambiguous 'half-true' claims under our documented binary research protocol, yielding 1,016 cleanly evaluated short political statements. We mapped 'false', 'pants-fire', and 'barely-true' to FAKE, and 'true' and 'mostly-true' to REAL, resulting in 556 actual FAKE and 460 actual REAL claims.
>
> Our confusion matrix reveals stark empirical findings:
> - The model produced 416 True Positives and 140 False Negatives on FAKE claims.
> - On REAL claims, it produced 376 False Positives and 84 True Negatives.
> - The total number of correct predictions is 416 plus 84, which equals 500 correct classifications out of 1,016, yielding an overall accuracy of 49.21%.
> - In total, the model predicted FAKE on 792 out of 1,016 claims—nearly 78% of the dataset—resulting in a low REAL recall of just 18.26%."

---

### Slide 8: Measured Metrics & Academic Honesty
- **Slide Title:** Measured Metrics & Academic Honesty
- **Subtitle:** Benchmark Results vs. Majority-Class Baseline
- **Allocated Time:** 4:00 – 4:35 (35 seconds)

#### Speaker Script:
> "The overall accuracy of the classifier on the evaluated LIAR test subset was **49.21%** (500 correct predictions out of 1,016), yielding a Macro F1 score of **0.4314**.
>
> To put this in proper scientific context: a naive majority-class heuristic that simply guesses 'FAKE' on every single claim without inspecting the text would achieve **54.72%** accuracy on this dataset (556 / 1,016). Our deep transformer model performed 5.51 percentage points *below* the naive majority baseline.
>
> In our research investigations during Milestone 6, we evaluated decision threshold tuning. Shifting the threshold from 0.50 down to 0.13 raised accuracy to 52.85%, which still remained below the majority baseline. Candidate model screening on validation data similarly showed that neither alternative model surpassed the validation majority baseline.
>
> It is essential to distinguish verified findings from hypotheses. The measured 49.21% accuracy, the absence of external evidence retrieval, and score uncalibration are verified empirical realities. Meanwhile, domain shift from undocumented training corpora remains a leading hypothesis rather than a proven cause, since the model author did not publish training data or preprocessing pipelines. Reporting these results transparently exemplifies the scientific integrity of our project."

---

### Slide 9: Summary & Future Roadmap
- **Slide Title:** Summary & Future Roadmap
- **Subtitle:** Engineering Achievements and the Path to Trustworthy AI
- **Allocated Time:** 4:35 – 5:00 (25 seconds)

#### Speaker Script:
> "To conclude, TruthLens successfully delivers a robust full-stack architecture with dual health routes (`/health` and `/api/health`), 15-second timeouts, and 100% automated test coverage across 73 backend tests and 12 frontend tests, verified offline and protected by GitHub Actions CI. Most importantly, it acts as a pioneer in trust transparency by openly communicating model limitations directly in the UI.
>
> For future research, we propose a three-point roadmap:
> 1. Implement Retrieval-Augmented Generation (RAG) to query fact-check databases (such as ClaimReview) before classifying claims.
> 2. Apply temperature scaling and Platt calibration on validation logits to align output scores with empirical accuracy distributions.
> 3. Integrate Explainable AI (XAI) token attribution heatmaps to visualize model attention for users.
>
> Thank you for your time, and I welcome any questions from the committee."

---

### Slide-by-Slide Timing Summary

| Slide # | Slide Title | Allotted Time | Cumulative Time |
|:-------:|:------------|:-------------:|:---------------:|
| **1** | Title & Technology Summary | 0:30 | 0:30 |
| **2** | Problem Context & Motivation | 0:30 | 1:00 |
| **3** | Project Goals & Scope | 0:30 | 1:30 |
| **4** | Engineering Design & System Architecture | 0:35 | 2:05 |
| **5** | Inference Pipeline & Mechanics | 0:35 | 2:40 |
| **6** | Live Application Demonstration | 0:40 | 3:20 |
| **7** | Rigorous Out-of-Domain Evaluation (Confusion Matrix) | 0:40 | 4:00 |
| **8** | Measured Metrics & Academic Honesty (49.21% vs 54.72%) | 0:35 | 4:35 |
| **9** | Summary & Future Roadmap | 0:25 | 5:00 |
| **Total** | **Complete Presentation** | **5:00 min** | **300 sec** |

---

### Key Numerical & Technical Fact Sheet for Defense Q&A

- **Model ID:** `dima806/fake-news-classifier` (Hugging Face)
- **Base Architecture:** `BertForSequenceClassification` (12 layers, 768 hidden, 12 heads, 110M params)
- **Max Tokenizer Length:** 512 tokens (BERT Cased WordPiece)
- **Health Check Routes:** `GET /health` and `GET /api/health` (parity routes; neither triggers premature model loading)
- **Input Validation:** 1 to 50,000 characters; rejects empty, whitespace-only, or solely invisible/zero-width formatting characters (Unicode categories `Cf`, `Cc`, `Cs`, `Cn`)
- **LIAR Benchmark Dataset:** William Yang Wang (ACL 2017), test split `test.tsv`
- **Total Test Rows:** 1,283 rows
- **Excluded 'half-true' Rows:** 267 rows
- **Evaluated Test Rows:** 1,016 rows
- **Class Ground Truth:** 556 FAKE (54.72%), 460 REAL (45.28%)
- **Model Predictions:** 792 FAKE (77.95%), 224 REAL (22.05%)
- **Confusion Matrix:**
  - Actual FAKE $\rightarrow$ Predicted FAKE: **416** (TP)
  - Actual FAKE $\rightarrow$ Predicted REAL: **140** (FN)
  - Actual REAL $\rightarrow$ Predicted FAKE: **376** (FP)
  - Actual REAL $\rightarrow$ Predicted REAL: **84** (TN)
- **Correct Predictions:** $416 + 84 = 500$
- **Evaluated Accuracy:** **49.21%** ($500 / 1,016 = 0.492125...$)
- **Majority-Class Baseline:** **54.72%** ($556 / 1,016 = 0.547244...$)
- **Macro F1 Score:** **0.4314**
- **REAL Class Metrics:** Precision = **0.3750** (84 / 224), Recall = **0.1826** (84 / 460), F1 = **0.2456**
- **FAKE Class Metrics:** Precision = **0.5253** (416 / 792), Recall = **0.7482** (416 / 556), F1 = **0.6172**
- **Threshold Sensitivity:** Shifting decision boundary from 0.50 to 0.13 yielded 52.85% accuracy (still below 54.72% majority baseline).
- **Research Transparency:** Softmax scores are mathematical label preferences, not calibrated probabilities of factual truth.
