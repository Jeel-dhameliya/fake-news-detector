# TruthLens — Milestone 6A: Classifier Diagnostic Audit

This diagnostic audit investigates the operational characteristics, error patterns, and contract integrity of the baseline classification model (`dima806/fake-news-classifier`) following its 49.21% accuracy on the LIAR benchmark test split.

---

## 1. Baseline Performance Reference

* **Model ID**: `dima806/fake-news-classifier`
* **Benchmark**: LIAR Benchmark Test Split (`test.tsv`, William Yang Wang, ACL 2017)
* **Dataset Source**: UCSB Original Archive (`https://www.cs.ucsb.edu/~william/data/liar_dataset.zip`, SHA-256: `611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7`)
* **Evaluation Scope**: 1,016 evaluated claims (267 `half-true` excluded from 1,283 total rows)
* **Operational Mapping**:
  * `FAKE`: `false` (250) + `pants-fire` (92) + `barely-true` (214) = 556
  * `REAL`: `true` (211) + `mostly-true` (249) = 460
* **Baseline Metrics**:
  * Accuracy: **49.21%** (500 / 1,016)
  * Macro F1: **0.4314** | Weighted F1: **0.4490**
  * FAKE (Positive): Precision **0.5253**, Recall **0.7482**, F1 **0.6172**
  * REAL: Precision **0.3750**, Recall **0.1826**, F1 **0.2456**
  * Confusion Matrix (Rows: Actual `[FAKE, REAL]`, Columns: Predicted `[FAKE, REAL]`):
    $$\begin{pmatrix} 416 & 140 \\ 376 & 84 \end{pmatrix}$$

---

## 2. Model Contract & Pipeline Verification

An audit of the production inference service (`backend/app/services/classifier.py`) and the evaluation script (`backend/scripts/evaluate_model.py`) confirmed exact contract parity:

| Contract Property | Production Classifier | Evaluation Script | Verification Status |
| :--- | :--- | :--- | :--- |
| **Model ID** | `dima806/fake-news-classifier` | `dima806/fake-news-classifier` | Identical |
| **Config `id2label`** | `{0: 'FAKE', 1: 'REAL'}` | `{0: 'FAKE', 1: 'REAL'}` | Identical |
| **Config `label2id`** | `None` (fallback parsed) | Inferred `{FAKE: 0, REAL: 1}` | Consistent |
| **Tokenizer Max Length** | 512 tokens | 512 tokens | Identical |
| **Truncation Policy** | `truncation=True` | `truncation=True` | Identical |
| **Inference Mode** | `torch.no_grad()` / `model.eval()` | `torch.inference_mode()` / `model.eval()` | Gradients disabled |
| **Hardware Device** | Apple Silicon `mps` (CPU fallback) | Apple Silicon `mps` (CPU fallback) | Identical |
| **Output Normalization** | `torch.softmax(logits, dim=-1)` | `torch.softmax` / `argmax(logits)` | Consistent |
| **Determinism** | Identical logits across runs | Identical logits across runs | Deterministic |

### Input Sanitization & Boundary Handling
* **Empty / Whitespace-only Input**: Rejected at API schema level with HTTP 422 (`MIN_INPUT_LENGTH=1`).
* **Maximum Input Length**: Capped at 50,000 characters by Pydantic validators.
* **Short Statements**: LIAR claims average ~18 words and do not exceed 512 tokens; zero claims experienced truncation.
* **Unicode & Punctuation**: Successfully tokenized without exceptions or NaN probabilities.

---

## 3. Empirical Error Breakdown & Statistical Analysis

Analysis was performed using the saved row-level artifact (`backend/evaluation_results/liar_predictions_compact.json`) across all 1,016 evaluated items without re-transcribing claim text.

### Class Distributions
* **Ground Truth**: 556 `FAKE` (54.72%) vs. 460 `REAL` (45.28%) — a mild 1.21:1 ratio.
* **Predictions**: 792 `FAKE` (77.95%) vs. 224 `REAL` (22.05%) — a 3.54:1 ratio.

### Error Asymmetry
* **Total Errors**: 516 out of 1,016 (50.79% overall error rate).
* **False-FAKE Errors (False Positives)**: **376** (72.87% of all errors).
  * Error rate on actual `REAL` statements: $376 / 460 = \mathbf{81.74\%}$.
* **False-REAL Errors (False Negatives)**: **140** (27.13% of all errors).
  * Error rate on actual `FAKE` statements: $140 / 556 = \mathbf{25.18\%}$.
* **Conclusion**: Errors are overwhelmingly concentrated in misclassifying truthful statements as `FAKE`.

### Fine-Grained Analysis by Original LIAR Label

| Original Label | Operational Ground Truth | Total Items | Predicted FAKE | % Predicted FAKE | Predicted REAL | % Predicted REAL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `barely-true` | `FAKE` | 214 | 167 | 78.04% | 47 | 21.96% |
| `false` | `FAKE` | 250 | 192 | 76.80% | 58 | 23.20% |
| `mostly-true` | `REAL` | 249 | 212 | **85.14%** | 37 | 14.86% |
| `pants-fire` | `FAKE` | 92 | 57 | 61.96% | 35 | 38.04% |
| `true` | `REAL` | 211 | 164 | **77.73%** | 47 | 22.27% |

#### Critical Observations from Raw Breakdown:
1. **Pervasive FAKE Prediction Rate**: Across all categories, the model predicted `FAKE` between 61.96% and 85.14% of the time.
2. **Inversion on `mostly-true`**: Actual `mostly-true` claims were predicted as `FAKE` at an 85.14% rate—higher than actual `false` (76.80%) or `pants-fire` (61.96%).
3. **Polar Extremes Analysis (`pants-fire` vs. `true`)**:
   Even if ambiguous or moderate categories (`barely-true`, `mostly-true`) are removed and only the polar extremes are evaluated:
   * Total polar items: $92\text{ (pants-fire)} + 211\text{ (true)} = 303$.
   * Correct predictions: $57\text{ (pants-fire } \rightarrow \text{ FAKE)} + 47\text{ (true } \rightarrow \text{ REAL)} = 104$.
   * Polar Accuracy: $104 / 303 = \mathbf{34.32\%}$.
   * The model predicted `FAKE` *more frequently* on pure `true` claims (77.73%) than on blatant `pants-fire` falsehoods (61.96%).
   * This proves that the poor performance is **not** an artifact of mapping `mostly-true` or `barely-true`.

---

## 4. Does Class Imbalance Explain the Results?

**No.** A numerical comparison against standard baselines confirms that class imbalance does not account for the observed metrics:

| Strategy | Decision Rule | Accuracy |
| :--- | :--- | :--- |
| **Majority Class Baseline** | Always predict `FAKE` | **54.72%** (556 / 1,016) |
| **Uniform Random Guess** | 50% `FAKE`, 50% `REAL` | **50.00%** |
| **Stratified Random Guess** | Predict according to priors ($0.5472^2 + 0.4528^2$) | **50.45%** |
| **Observed Baseline Model** | BERT sequence classification | **49.21%** (500 / 1,016) |

* The model achieves lower accuracy than a zero-intelligence majority class classifier (54.72%) and lower accuracy than pure chance (50.00%).
* The failure mode is an active misclassification of truthful content driven by a severe tendency to predict `FAKE` (77.95% prediction rate).

---

## 5. Epistemic Separation of Audit Findings

To maintain scientific integrity, findings are categorized into verified facts, plausible hypotheses, and unknowns:

### Verified Facts
1. The production pipeline and evaluation script share identical label mappings (`{0: 'FAKE', 1: 'REAL'}`), tokenizers, and truncation parameters.
2. The model exhibits a 77.95% bias toward predicting `FAKE` on LIAR test statements.
3. The model's error rate on actual `REAL` claims is 81.74%, accounting for 72.87% of all misclassifications.
4. On polar extremes (`pants-fire` vs. `true`), model accuracy drops further to 34.32%.
5. Model confidence scores are saturated (frequently $\ge 0.99$), reflecting uncalibrated logit separation rather than reliable veracity probabilities.

### Plausible Hypotheses
1. **Task / Format Domain Shift**: The model may have been trained on full-length news articles containing journalistic structure, bylines, and narrative context. Short, isolated political soundbites lack these signals and may trigger false-positive `FAKE` predictions.
2. **Vocabulary Distribution Shift**: Words prevalent in political discourse (e.g., policy terms, legislative references) may be disproportionately represented in the model's training-set `FAKE` class.
3. **Threshold Distortion**: The default argmax decision boundary (logit differential $> 0$) may be misaligned with out-of-domain short-text distributions.

### Unknowns (Questions Evidence Cannot Answer)
1. **Training Data Composition**: The training dataset, source distribution, and fine-tuning parameters for `dima806/fake-news-classifier` are not documented by the model author.
2. **Pretraining / Benchmark Overlap**: Whether any portion of PolitiFact or the LIAR benchmark appeared in pretraining data cannot be confirmed or refuted from weights alone.
3. **Causal Attribution**: Without access to training ablation runs, whether short text length, specific vocabulary tokens, or label imbalance during training caused the `FAKE` skew remains unproven.

---

## 6. Recommended Next Diagnostic Experiments

Before replacing or fine-tuning models, the following non-destructive experiments are recommended:

1. **In-Domain Benchmark Evaluation**: Evaluate `dima806/fake-news-classifier` on a full-length news article test set (e.g., WELFake or ISOT) to determine whether accuracy recovers when text format matches typical news articles.
2. **Statement Length Sensitivity Probe**: Group LIAR statements into length deciles (character and token counts) to determine whether prediction bias correlates with input length.
3. **Decision Threshold Calibration**: Analyze ROC and Precision-Recall curves on LIAR validation split to determine whether adjusting the classification threshold from 0.5 can restore balanced precision and recall.
4. **Logit and Temperature Analysis**: Measure raw logit distributions across classes to quantify logit saturation and evaluate temperature scaling calibration.
