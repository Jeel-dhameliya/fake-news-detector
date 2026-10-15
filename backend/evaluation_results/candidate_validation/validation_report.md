# TruthLens — Milestone 6D: Controlled Candidate Model Evaluation

This evaluation report benchmarks candidate models against the LIAR benchmark validation split (`valid.tsv`) to determine whether any model warrants further consideration before proceeding to test-set evaluations.

---

## 1. Evaluation Protocol & Accounting

* **Split Filename**: `LIAR validation split (valid.tsv)` (strictly `valid.tsv`; `test.tsv` was not accessed)
* **Archive Source URL**: `https://www.cs.ucsb.edu/~william/data/liar_dataset.zip`
* **Archive SHA-256 Checksum**: `611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7` (verified)
* **Row Accounting**:
  * Total rows: **1284**
  * Excluded (`half-true`): **248**
  * Evaluated rows: **1036**
  * Ground truth: **616 FAKE** (59.46%) vs. **420 REAL** (40.54%)

---

## 2. Predeclared Screening Criteria

To prevent selective bias or shifting standards, candidate models must satisfy four predeclared criteria on the validation split:

1. **Accuracy**: Must exceed the validation majority-class baseline of **59.46%** (616 / 1,036).
2. **REAL Class Recall**: Must be at least **0.4000** (40%).
3. **Macro F1**: Must be at least **0.5500**.
4. **Polar Extreme Discrimination**: Accuracy on `pants-fire` vs. `true` statements must exceed **50.00%**.

---

## 3. Comparative Validation Metrics

| Evaluation Metric | Majority Class Baseline | Production Baseline (`dima806`) | Candidate 1 (`naheelkk`) | Candidate 2 (`Vs2882`) |
| :--- | :---: | :---: | :---: | :---: |
| **Model ID / Strategy** | Always predict FAKE | `dima806/fake-news-classifier` | `naheelkk/fake-news-bert-liar` | `Vs2882/liar_binaryclassifier_roberta_base` |
| **Architecture** | Heuristic | BERT-base (cased) | BERT-base (uncased) | RoBERTa-base |
| **License** | N/A | Apache 2.0 | Apache 2.0 | MIT |
| **Validation Accuracy** | 59.46% | 54.54% | 53.76% | 56.18% |
| **Balanced Accuracy** | 0.5000 | 0.4874 | 0.5976 | 0.5671 |
| **Macro F1** | 0.3729 | 0.4595 | 0.5176 | 0.5590 |
| **FAKE Precision / Recall** | 0.5946 / 1.0000 | 0.5870 / 0.7938 | 0.8278 / 0.2808 | 0.6614 / 0.5390 |
| **REAL Precision / Recall** | 0.0000 / 0.0000 | 0.3744 / 0.1810 | 0.4643 / 0.9143 | 0.4682 / 0.5952 |
| **Polar Accuracy (`pants-fire` vs `true`)** | 32.86% | 37.54% | 72.63% | 60.70% |
| **Predicted Class Distribution (FAKE / REAL)** | 1036 / 0 | 833 / 203 | 209 / 827 | 502 / 534 |

---

## 4. Confusion Matrices (Rows: Actual [FAKE, REAL], Columns: Predicted [FAKE, REAL])

* **Production Baseline (`dima806`)**:
  `[[489, 127], [344, 76]]`
* **Candidate 1 (`naheelkk`)**:
  `[[173, 443], [36, 384]]`
* **Candidate 2 (`Vs2882`)**:
  `[[332, 284], [170, 250]]`

---

## 5. Screening Criteria Evaluation

| Predeclared Criterion | Threshold | Baseline (`dima806`) | Candidate 1 (`naheelkk`) | Candidate 2 (`Vs2882`) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy > Majority** | > 59.46% | ❌ FAIL | ❌ FAIL | ❌ FAIL |
| **REAL Recall >= 40%** | >= 40.00% | ❌ FAIL | ✅ PASS | ✅ PASS |
| **Macro F1 >= 0.55** | >= 0.5500 | ❌ FAIL | ❌ FAIL | ✅ PASS |
| **Polar Accuracy > 50%** | > 50.00% | ❌ FAIL | ✅ PASS | ✅ PASS |
| **OVERALL OUTCOME** | **All 4 Pass** | **❌ FAIL** | **❌ FAIL** | **❌ FAIL** |

---

## 6. Analytical Findings & Structural Conclusions

1. **Baseline Reproduction**:
   The production model `dima806/fake-news-classifier` achieved **54.54%** accuracy and **0.4595** Macro F1 on validation, with a REAL recall of only **18.10%**. This closely mirrors its 49.21% test accuracy and confirms its severe bias toward predicting FAKE across splits.

2. **Candidate Comparison**:
   * **Candidate 1 (`naheelkk/fake-news-bert-liar`)**: Accuracy of **53.76%**, Macro F1 of **0.5176**, REAL recall of **91.43%**.
   * **Candidate 2 (`Vs2882/liar_binaryclassifier_roberta_base`)**: Accuracy of **56.18%**, Macro F1 of **0.5590**, REAL recall of **59.52%**.

3. **Screening Verdict**:
   Neither candidate met all four predeclared screening criteria. In accordance with strict experimental protocol, neither candidate is recommended for production replacement or further test-set evaluation.

4. **Measured Findings & Empirical Scope**:
   * *Measured Finding*: On this validation split, neither candidate model achieved higher accuracy than the simple majority-class baseline (59.46%), with Candidate 1 reaching 53.76% and Candidate 2 reaching 56.18%.
   * *Hypothesis*: The performance limitation observed on short political claims may stem from evaluating claims without external contextual evidence, but establishing this causal mechanism requires retrieval-augmented experiments rather than text classification alone.
