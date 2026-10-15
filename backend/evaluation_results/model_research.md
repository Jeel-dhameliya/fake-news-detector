# TruthLens — Milestone 6C: Research Better-Suited Classification Models

This research document evaluates the structural causes of `dima806/fake-news-classifier`'s failure on short statements, defines the taxonomy of truth-detection tasks, audits candidate models from public repositories against strict provenance standards, and proposes a disciplined, validation-only protocol for any future model benchmarking.

---

## 1. Project Baseline & Problem Formulation

### 1.1 Baseline Configuration and Evidence
* **Model ID**: `dima806/fake-news-classifier`
* **Architecture**: Fine-tuned BERT base (`BertForSequenceClassification`, uncased, 12 layers, 768 hidden dimensions, 12 attention heads, ~109.5M parameters).
* **Tokenizer & Context**: WordPiece uncased; maximum position embeddings: 512 tokens.
* **Production Label Schema**: Verified `id2label = {0: "FAKE", 1: "REAL"}` (enforced in `backend/app/config.py`).
* **Input Pipeline Limits**: API accepts 1 to 50,000 characters; tokenizer truncates at 512 tokens.
* **Measured Benchmark Performance** (LIAR test split, 1,016 evaluated claims, 267 `half-true` excluded):
  * Test Accuracy: **49.21%** (worse than 54.72% majority-class baseline and 50.00% uniform chance).
  * Macro F1: **0.4314** | Weighted F1: **0.4490**
  * REAL Class Recall: **0.1826** (84 / 460 correctly identified; 81.74% of true claims misclassified as FAKE).
  * Extreme Polarization: Test accuracy on `pants-fire` vs. `true` is only **34.32%**.
  * Threshold Tuning Limit: Exhaustive search in Milestone 6B revealed that adjusting decision threshold $\theta \in [0.01, 0.50]$ yielded a maximum test accuracy of only **49.41%** (a trivial +2 net correct predictions out of 1,016 claims) due to severe logit saturation (>85% of claims assigned $P(\text{REAL}) < 0.002$ or $> 0.98$).

### 1.2 Core Taxonomy: Three Distinct Problem Formulations
A critical insight from the diagnostic audit is that "fake news detection" is not a monolithic NLP task. Conflating distinct tasks leads directly to the failure observed with `dima806`:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TASK TAXONOMY COMPARISON                               │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│ 1. Full-Article Fake News     │ 2. Short-Claim Veracity       │ 3. Evidence-Based      │
│    Classification             │    Classification             │    Fact Verification   │
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ • Input: Full news articles   │ • Input: Isolated statement   │ • Input: Claim +       │
│   (headline + body, 300-2000w)│   (soundbite, 10-30 words)    │   retrieved evidence   │
│ • Signals: Sensationalist     │ • Signals: Plausibility,      │ • Signals: Textual     │
│   syntax, informal diction,   │   speaker metadata, semantic  │   entailment, support, │
│   byline/dateline patterns,   │   coherence (surface only).   │   contradiction against│
│   journalistic conventions.   │                               │   authoritative text.  │
│ • Datasets: Kaggle ISOT,      │ • Datasets: LIAR, LIAR-2,     │ • Datasets: FEVER,     │
│   WELFake, FakeNewsNet.       │   PolitiFact statements.      │   HoVer, SciFact.      │
│ • Limitation: Cannot verify   │ • Limitation: Lacks grounding;│ • Limitation: Requires │
│   factual assertions; exploits│   factual truth cannot be     │   retrieval pipeline & │
│   stylistic correlation.      │   deduced from syntax alone.  │   knowledge corpus.    │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┤
```

1. **Full-Article Fake News Classification** (`dima806`'s likely training regime):
   * Models learn journalistic style, vocabulary distribution, source artifacts, and sensationalist phrasing.
   * When fed an isolated 15-word political soundbite, the model experiences extreme out-of-domain distribution shift. The lack of standard journalistic context triggers high-confidence default predictions (`FAKE`).
2. **Short Political Claim Veracity Classification** (LIAR benchmark task):
   * Models evaluate standalone political soundbites without external evidence.
   * While structurally matched to the LIAR input format, historical benchmarks (e.g., Wang 2017) demonstrate that purely textual models struggle on this task (typically achieving 55–62% binary accuracy) because whether a politician told the truth usually depends on real-world facts not present in the sentence itself.
3. **Evidence-Based Fact Verification** (FEVER / Real-World Fact Checking):
   * Models assess the logical relationship between a claim and a retrieved body of verified evidence (Natural Language Inference: Entailment, Contradiction, Neutral).
   * This is epistemically rigorous, but requires an external retrieval engine and indexed knowledge base (e.g., Wikipedia or trusted news archives), which changes the system architecture from a single sequence classifier to a retrieval-augmented verification pipeline.

---

## 2. Candidate Model Audit & Comparative Analysis

To identify models better aligned with short-text claim classification, three candidate models hosted on the Hugging Face Hub were investigated. Per project rules, no weights were downloaded, and no candidates are presumed superior without empirical validation.

---

### Candidate 1: `naheelkk/fake-news-bert-liar`
* **Hugging Face Model ID**: `naheelkk/fake-news-bert-liar`
* **Model Card URL**: [https://huggingface.co/naheelkk/fake-news-bert-liar](https://huggingface.co/naheelkk/fake-news-bert-liar)
* **Intended Task**: Binary text classification for fake vs. real political statements.
* **Model Architecture**:
  * Architecture: `BertForSequenceClassification` (`bert-base-uncased`)
  * Parameters: ~109.5M parameters (12 hidden layers, 768 hidden dimensions, 12 attention heads)
  * Memory/Compute: ~420 MB float32 weights; lightweight inference on CPU and Apple Silicon MPS.
* **Documented Training Details**:
  * Base Model: `bert-base-uncased`
  * Training Dataset: `ucsbai/liar` dataset (fact-checked political statements).
  * Hyperparameters: 3 epochs, batch size 16, learning rate 1e-5, max sequence length 128.
  * Training Label Grouping (from Model Card):
    * **REAL**: `true`, `mostly-true`, `half-true`
    * **FAKE**: `barely-true`, `false`, `pants-on-fire`
* **Published Metrics** (Reported on author's test split):
  * Test Accuracy: **60.86%**
  * F1 Score: **71.80%** | Precision: **60.54%** | Recall: **88.22%**
  * Out-of-domain article check: 80% reported accuracy on Reuters/BBC style articles.
* **License**: Apache 2.0 (permissive, open for commercial and research use).
* **Alignment with TruthLens Benchmark**:
  * *Strengths*: Directly trained on short political soundbites from the same underlying PolitiFact corpus as LIAR. Sequence length (max 128) aligns with short claims.
  * *Discrepancies / Risks*:
    1. **Operational Mapping Divergence**: The author mapped `half-true` to `REAL`, whereas our benchmark protocol excludes `half-true`.
    2. **Missing `id2label` in `config.json`**: Inspecting `config.json` confirms `id2label` is not set (defaults to `{"0": "LABEL_0", "1": "LABEL_1"}`). The model card documentation explicitly defines:
       ```python
       label = "REAL" if prediction == 1 else "FAKE"
       ```
       An adapter or explicit mapping (`{0: "FAKE", 1: "REAL"}`) is required to prevent label inversion.

---

### Candidate 2: `Vs2882/liar_binaryclassifier_roberta_base`
* **Hugging Face Model ID**: `Vs2882/liar_binaryclassifier_roberta_base`
* **Model Card URL**: [https://huggingface.co/Vs2882/liar_binaryclassifier_roberta_base](https://huggingface.co/Vs2882/liar_binaryclassifier_roberta_base)
* **Intended Task**: Single-label binary sequence classification of political statements.
* **Model Architecture**:
  * Architecture: `RobertaForSequenceClassification` (`roberta-base`)
  * Parameters: ~124.6M parameters (12 hidden layers, 768 hidden dimensions, 12 attention heads, max position embeddings 514).
  * Memory/Compute: ~500 MB float32 weights; fully compatible with existing PyTorch/MPS infrastructure.
* **Documented Training Details**:
  * Base Model: `roberta-base`
  * Training Dataset: Hugging Face `liar` dataset via `transformers.Trainer`.
  * Hyperparameters: 5 epochs, Adam optimizer (lr 3e-6, $\beta_1=0.9, \beta_2=0.999$, $\epsilon=1\text{e-}8$), train/eval batch size 8, linear learning rate decay.
* **Published Metrics** (from automated Trainer log metadata):
  * Validation Accuracy: **57.70%** (Validation Loss: 0.6621 at epoch 5).
  * Test Accuracy (metadata tag): **57.70%** (`0.5770065075921909`).
* **License**: MIT (permissive, open-source).
* **Alignment with TruthLens Benchmark**:
  * *Strengths*:
    1. Full reproducibility metadata: exact training loss steps, learning rates, and Trainer logs recorded.
    2. Explicit label configuration: `config.json` defines `id2label = {"0": "false", "1": "true"}` and `label2id = {"false": 0, "true": 1}`, perfectly mirroring our binary schema (`0 = FAKE`, `1 = REAL`).
    3. RoBERTa architecture benefits from larger byte-pair BPE vocabulary and dynamic masking.
  * *Discrepancies / Risks*:
    1. The model card does not explicitly state whether `half-true` was dropped or mapped into one of the classes during training. Validation evaluation is necessary to verify class calibration.
    2. Headline accuracy (57.70%) is modest, illustrating the inherent ceiling of text-only claim veracity classification.

---

### Candidate 3: `Dzeniks/roberta-fact-check`
* **Hugging Face Model ID**: `Dzeniks/roberta-fact-check`
* **Model Card URL**: [https://huggingface.co/Dzeniks/roberta-fact-check](https://huggingface.co/Dzeniks/roberta-fact-check)
* **Intended Task**: Evidence-based fact verification (pair classification: `[claim, evidence]`).
* **Model Architecture**:
  * Architecture: `RobertaForSequenceClassification` (`roberta-base`)
  * Parameters: ~124.6M parameters.
  * Memory/Compute: ~500 MB float32 weights.
* **Documented Training Details**:
  * Base Model: `roberta-base`
  * Training Datasets: FEVER (Fact Extraction and VERification) and HoVer (multi-hop fact verification), plus manually curated pairs.
  * Input Format: `tokenizer.encode_plus(claim, evidence)`
  * Output Schema: Label `0`: Supports, Label `1`: Refutes.
* **Published Metrics**: General multi-dataset validation reported; specific test split accuracy depends on evidence retrieval quality (typically 75–85% on FEVER with oracle evidence).
* **License**: MIT.
* **Alignment with TruthLens Benchmark**:
  * *Strengths*: Represents the gold-standard epistemic approach to fact-checking (evaluating veracity conditional on factual evidence rather than predicting from assertion text alone).
  * *Critical Architectural Incompatibility*:
    1. **Input Interface**: Requires a paired evidence document for inference (`model(claim, evidence)`). TruthLens currently provides only standalone statements (`model(claim)`).
    2. Feeding an empty string or the claim itself as evidence violates the model's inductive bias and produces undefined predictions.
    3. While unsuitable as a direct drop-in replacement for the single-input classifier, this candidate establishes the technical blueprint for any future Milestone exploring Retrieval-Augmented Generation (RAG) or evidence-grounded verification.

---

### Comparative Summary Matrix

| Evaluation Dimension | Existing Production Baseline | Candidate 1 (`naheelkk`) | Candidate 2 (`Vs2882`) | Candidate 3 (`Dzeniks`) |
| :--- | :--- | :--- | :--- | :--- |
| **Hugging Face ID** | `dima806/fake-news-classifier` | `naheelkk/fake-news-bert-liar` | `Vs2882/liar_binaryclassifier_roberta_base` | `Dzeniks/roberta-fact-check` |
| **Model Card URL** | [HF Hub](https://huggingface.co/dima806/fake-news-classifier) | [HF Hub](https://huggingface.co/naheelkk/fake-news-bert-liar) | [HF Hub](https://huggingface.co/Vs2882/liar_binaryclassifier_roberta_base) | [HF Hub](https://huggingface.co/Dzeniks/roberta-fact-check) |
| **Task Category** | Full-article fake news | Short political claim veracity | Short political claim veracity | Evidence-based fact verification |
| **Base Architecture** | BERT-base (uncased) | BERT-base (uncased) | RoBERTa-base | RoBERTa-base |
| **Parameter Count** | ~110M (~420 MB) | ~110M (~420 MB) | ~125M (~500 MB) | ~125M (~500 MB) |
| **License** | Not documented | Apache 2.0 | MIT | MIT |
| **Input Format** | Single text string | Single text string | Single text string | Paired `[claim, evidence]` |
| **Compatible with Drop-in API?** | Current (`/predict`) | **Yes** (drop-in) | **Yes** (drop-in) | **No** (requires retrieval engine) |
| **Config `id2label` State** | Hardcoded verified fallback | Default `LABEL_0, LABEL_1` | Explicit `{"0": "false", "1": "true"}` | Missing in config |
| **Documented Target Dataset** | Undocumented (likely article corpus) | `ucsbai/liar` | `ucsbai/liar` | FEVER / HoVer |
| **Reported Benchmark Metric** | **49.21%** test accuracy (LIAR) | **60.86%** test accuracy (LIAR) | **57.70%** val/test accuracy (LIAR)| ~80% with oracle evidence |
| **Directly Comparable?** | Measured locally | No (different split/mapping) | No (different split/mapping) | No (different task & inputs) |

> [!WARNING]
> Reported metrics on Hugging Face model cards cannot be directly compared against the TruthLens 49.21% baseline. Candidate 1 included `half-true` in the `REAL` class, while Candidate 2's specific handling of `half-true` is unverified. Any legitimate comparison requires evaluating both models under the exact same operational protocol on local data.

---

## 3. Recommended Controlled Experiment Protocol

If a future milestone authorizes downloading and evaluating replacement candidates, the following experimental protocol must be strictly adhered to:

### 3.1 Strict Separation of Splits
1. **Validation-Only Selection**:
   * All candidate evaluation and threshold tuning must be executed strictly on the **validation split** (`valid.tsv`, 1,036 evaluated claims).
   * Neither Candidate 1 nor Candidate 2 may be evaluated on the test split during the selection phase.
2. **Held-Out Test Set**:
   * The test split (`test.tsv`, 1,016 evaluated claims) remains locked.
   * Only the single, winning candidate chosen from validation performance will be run once on the test split for final confirmation.

### 3.2 Fixed Operational Mapping
To ensure scientific consistency with Milestones 5 and 6, the evaluation protocol must remain identical:
* **Source**: Official UCSB LIAR archive (`https://www.cs.ucsb.edu/~william/data/liar_dataset.zip`, SHA-256: `611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7`).
* **Evaluation Scope**:
  * Validation: 1,036 claims (248 `half-true` excluded from 1,284 total rows).
  * Class Breakdown: 616 `FAKE` (59.46%), 420 `REAL` (40.54%).
* **Ground Truth Mapping**:
  * `FAKE`: `false`, `pants-fire`, `barely-true`
  * `REAL`: `true`, `mostly-true`
  * Excluded: `half-true`

### 3.3 Evaluation Metrics
For every evaluated model, report:
1. **Accuracy** and **Balanced Accuracy** (to account for the 59.5% validation class skew).
2. **Macro-averaged F1** and **Weighted F1**.
3. **Per-class Precision, Recall, and F1** for both `FAKE` and `REAL`.
4. **Confusion Matrix** (Rows: Actual `[FAKE, REAL]`, Columns: Predicted `[FAKE, REAL]`).
5. **Polar Extreme Accuracy** (`pants-fire` vs. `true`).
6. **Inference Latency & Throughput** (p50, p95 latency on MPS / CPU).

### 3.4 Pre-Declared Selection & Adoption Criteria
A candidate should be considered for potential production replacement **only if it satisfies all of the following criteria on the validation split**:
1. **Exceeds the Majority-Class Baseline**: Must achieve Validation Accuracy $> 59.46\%$ (the trivial baseline of predicting all `FAKE`).
2. **Demonstrates Balanced Classification**:
   * REAL recall $\ge 0.40$ (baseline was an abysmal 0.1826 on test, 0.1690 on validation).
   * Macro F1 $\ge 0.5500$ (baseline was 0.4314).
3. **Polar Discrimination**: Must achieve $> 50\%$ accuracy on the subset of polar extremes (`pants-fire` vs. `true`).
4. **Architectural & License Parity**:
   * Permissive license (Apache 2.0 or MIT).
   * Standard SequenceClassification pipeline compatible with existing FastAPI lazy-loading singleton.
   * Latency under 100 ms per claim on Apple Silicon MPS.

If no candidate meets these minimum thresholds on validation data, the conclusion must state that standalone sequence classification without external evidence retrieval is insufficient for political claim verification.

---

## 4. Unresolved Uncertainties & Research Limitations

1. **Epistemic Limit of Sentence-Level Classifiers**:
   A short political claim (e.g., *"The state unemployment rate dropped by 1.2% in June"*) contains no inherent linguistic hallmarks of deceit. Its truth value is determined by external labor statistics. Therefore, any pure NLP classifier predicting veracity from claim text alone is learning statistical proxies (topic biases, speaker framing, lexical associations) rather than verifying factual truth.
2. **Dataset Annotation Inconsistencies**:
   The boundaries between `barely-true`, `half-true`, and `mostly-true` in human fact-checking annotations involve subjective editorial judgment. Different candidate models may have adopted varying binarization strategies during fine-tuning.
3. **Licensing Constraints of Base Weights**:
   While the community fine-tuned adapters carry Apache-2.0 or MIT licenses, commercial deployment requires verifying that underlying base checkpoints (BERT / RoBERTa) and any intermediate fine-tuning data remain compliant with production constraints.

---

## 5. Recommended Next Action

Do not alter production code or baseline evaluation reports. When authorized by a dedicated evaluation milestone:
1. Implement a non-destructive evaluation script `backend/scripts/evaluate_candidates.py` that loads candidates in an isolated environment.
2. Run validation-only evaluation for `naheelkk/fake-news-bert-liar` and `Vs2882/liar_binaryclassifier_roberta_base` against `valid.tsv`.
3. Compare both candidates against the validation baseline before making any decision regarding the held-out test split.
