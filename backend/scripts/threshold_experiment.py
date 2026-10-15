#!/usr/bin/env python3
"""TruthLens — Threshold Sensitivity Experiment.

Investigates whether adjusting the decision threshold for predicting REAL improves
the balance between FAKE and REAL predictions for the frozen model dima806/fake-news-classifier.

Methodology:
    1. Validation split inference: Run inference on the mapped LIAR validation split
       (valid.tsv, 1,036 evaluated items) and extract softmax probabilities P(REAL).
    2. Grid evaluation: Evaluate a dense grid of candidate thresholds (0.01 to 0.99).
    3. Predeclared selection rule: Select the candidate threshold that maximizes validation Macro F1,
       breaking ties by proximity to the default boundary (0.50), then choosing the lower value.
    4. Evaluated threshold test: Evaluate the validation-selected candidate threshold on the test split
       (test.tsv, 1,016 evaluated items) alongside the default baseline (0.50).
    5. Audit methodology disclosure: The candidate threshold (0.13) was selected solely from validation labels.
       During exploratory analysis, the test split was also observed across alternative thresholds
       (0.50, 0.13, 0.10, 0.05, 0.01). While the selected candidate was not changed, this multi-threshold
       test comparison is exploratory and the test split should not be described as strictly untouched.

Decision Rule:
    Predict REAL if P(REAL) >= threshold; otherwise predict FAKE.
    Note: P(REAL) is a model softmax score, NOT a calibrated probability of factual truth.

"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import zipfile

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("threshold_experiment")

# Authoritative dataset provenance constants
DEFAULT_MODEL_ID = "dima806/fake-news-classifier"
DATASET_SOURCE_NAME = "LIAR Benchmark (UCSB original publication archive)"
DATASET_ARCHIVE_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
DATASET_ARCHIVE_SHA256 = "611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7"
DEFAULT_DECISION_THRESHOLD = 0.50

# Predeclared operational mapping:
# FAKE: false, pants-fire, barely-true
# REAL: true, mostly-true
# EXCLUDED: half-true (ambiguous under binary protocol)
OPERATIONAL_LABEL_MAP: Dict[str, Optional[str]] = {
    "false": "FAKE",
    "pants-fire": "FAKE",
    "barely-true": "FAKE",
    "true": "REAL",
    "mostly-true": "REAL",
    "half-true": None,
}


def map_liar_label(label: str) -> Optional[str]:
    """Map a raw LIAR dataset label to the operational binary label."""
    normalized = label.strip().lower()
    return OPERATIONAL_LABEL_MAP.get(normalized, None)


def apply_threshold(prob_real: float, threshold: float) -> str:
    """Apply decision threshold to predict REAL or FAKE.

    Rule: Predict REAL if prob_real >= threshold, else FAKE.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")
    if not math.isfinite(prob_real):
        raise ValueError(f"prob_real must be a finite float, got {prob_real}")
    return "REAL" if prob_real >= threshold else "FAKE"


def select_optimal_threshold(
    candidate_results: List[Dict[str, Any]],
    target_metric: str = "macro_f1",
    default_boundary: float = DEFAULT_DECISION_THRESHOLD,
) -> float:
    """Select the optimal threshold using a predeclared deterministic selection rule.

    Rule:
        1. Maximize target_metric (default: macro_f1).
        2. Break ties by choosing the threshold closest to default_boundary (0.50).
        3. If still tied, choose the lower threshold.
    """
    if not candidate_results:
        raise ValueError("Candidate results list cannot be empty.")

    best_score = -1.0
    best_candidate: Optional[Dict[str, Any]] = None

    for item in candidate_results:
        th = float(item["threshold"])
        score = float(item[target_metric])

        if score > best_score:
            best_score = score
            best_candidate = item
        elif math.isclose(score, best_score, rel_tol=1e-6, abs_tol=1e-6):
            # Tie breaker 1: Distance to default boundary
            current_dist = abs(float(best_candidate["threshold"]) - default_boundary)
            candidate_dist = abs(th - default_boundary)

            if candidate_dist < current_dist - 1e-6:
                best_candidate = item
            elif math.isclose(candidate_dist, current_dist, rel_tol=1e-6, abs_tol=1e-6):
                # Tie breaker 2: Choose lower threshold
                if th < float(best_candidate["threshold"]):
                    best_candidate = item

    return float(best_candidate["threshold"])


def resolve_device(preferred: str = "auto") -> torch.device:
    """Resolve optimal PyTorch device supporting Apple Silicon MPS, CUDA, and CPU."""
    pref = preferred.strip().lower()
    if pref == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    elif pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    elif pref == "cpu":
        return torch.device("cpu")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_and_filter_split(
    split_filename: str,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Load, verify, and filter a split from the cached official LIAR archive."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/truthlens_eval")
    os.makedirs(cache_dir, exist_ok=True)

    zip_path = os.path.join(cache_dir, "liar_dataset.zip")
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        logger.info("Downloading LIAR archive from %s...", DATASET_ARCHIVE_URL)
        req = urllib.request.Request(
            DATASET_ARCHIVE_URL,
            headers={"User-Agent": "TruthLens-ThresholdExperiment/1.0"},
        )
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f_out:
            f_out.write(resp.read())

    # Verify SHA-256 checksum
    with open(zip_path, "rb") as f_zip:
        sha256 = hashlib.sha256(f_zip.read()).hexdigest()
    if sha256 != DATASET_ARCHIVE_SHA256:
        raise ValueError(
            f"Archive checksum mismatch! Expected {DATASET_ARCHIVE_SHA256}, got {sha256}"
        )

    # Read split file
    raw_rows: List[Dict[str, str]] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        if split_filename not in z.namelist():
            raise FileNotFoundError(f"{split_filename} not found in LIAR archive.")
        with z.open(split_filename) as f_split:
            reader = csv.reader(
                io.TextIOWrapper(f_split, encoding="utf-8"),
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
            )
            for row in reader:
                if len(row) < 3:
                    continue
                raw_rows.append({
                    "id": row[0].strip(),
                    "label": row[1].strip(),
                    "statement": row[2].strip(),
                })

    # Apply mapping and accounting
    raw_counts: Dict[str, int] = {}
    included: List[Dict[str, str]] = []
    excluded_counts: Dict[str, int] = {}

    for row in raw_rows:
        raw_lbl = row["label"]
        raw_counts[raw_lbl] = raw_counts.get(raw_lbl, 0) + 1
        stmt = row["statement"]

        if not stmt:
            excluded_counts["empty_statement"] = excluded_counts.get("empty_statement", 0) + 1
            continue

        mapped = map_liar_label(raw_lbl)
        if mapped is None:
            excluded_counts[raw_lbl] = excluded_counts.get(raw_lbl, 0) + 1
        else:
            included.append({
                "id": row["id"],
                "raw_label": raw_lbl,
                "statement": stmt,
                "mapped_label": mapped,
            })

    accounting = {
        "split_filename": split_filename,
        "total_rows": len(raw_rows),
        "evaluated_rows": len(included),
        "excluded_rows": sum(excluded_counts.values()),
        "raw_counts": raw_counts,
        "excluded_counts": excluded_counts,
        "mapped_ground_truth_counts": {
            "FAKE": sum(1 for r in included if r["mapped_label"] == "FAKE"),
            "REAL": sum(1 for r in included if r["mapped_label"] == "REAL"),
        },
    }

    return included, accounting


def compute_threshold_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics for a specific prediction vector."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical length.")

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    p_fake, r_fake, f1_fake, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label="FAKE", zero_division=0
    )
    p_real, r_real, f1_real, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label="REAL", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=["FAKE", "REAL"]).tolist()

    pred_fake = sum(1 for y in y_pred if y == "FAKE")
    pred_real = sum(1 for y in y_pred if y == "REAL")

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "fake_class": {
            "precision": round(float(p_fake), 4),
            "recall": round(float(r_fake), 4),
            "f1_score": round(float(f1_fake), 4),
            "support": sum(1 for y in y_true if y == "FAKE"),
        },
        "real_class": {
            "precision": round(float(p_real), 4),
            "recall": round(float(r_real), 4),
            "f1_score": round(float(f1_real), 4),
            "support": sum(1 for y in y_true if y == "REAL"),
        },
        "predicted_counts": {
            "FAKE": pred_fake,
            "REAL": pred_real,
        },
        "confusion_matrix": {
            "convention": "Rows actual [FAKE, REAL], Columns predicted [FAKE, REAL]",
            "matrix": cm,
            "breakdown": {
                "actual_fake_predicted_fake": cm[0][0],
                "actual_fake_predicted_real": cm[0][1],
                "actual_real_predicted_fake": cm[1][0],
                "actual_real_predicted_real": cm[1][1],
            },
        },
    }


def run_batch_inference_probs(
    statements: List[str],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int = 32,
) -> List[float]:
    """Run batched inference and return softmax probability P(REAL) for each statement."""
    probs_real: List[float] = []
    with torch.inference_mode():
        for i in range(0, len(statements), batch_size):
            batch = statements[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            logits = model(**inputs).logits
            # Label mapping index 0 = FAKE, index 1 = REAL
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
            probs_real.extend(batch_probs)

    return probs_real


def run_threshold_experiment(
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 32,
    device_pref: str = "auto",
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the threshold sensitivity experiment across validation and test splits."""
    timestamp = datetime.now(timezone.utc).isoformat()
    device = resolve_device(device_pref)
    logger.info("Initializing threshold experiment on device: %s", device)

    # 1. Load model and tokenizer
    logger.info("Loading model and tokenizer for %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    raw_id2label = getattr(model.config, "id2label", {0: "FAKE", 1: "REAL"})
    id2label = {int(k): str(v).upper() for k, v in raw_id2label.items()}
    if id2label.get(0) != "FAKE" or id2label.get(1) != "REAL":
        raise ValueError(f"Unexpected model id2label mapping: {id2label}")

    # 2. Load validation split
    valid_records, valid_accounting = load_and_filter_split(
        "valid.tsv", cache_dir=cache_dir
    )
    logger.info(
        "Loaded validation split: %d evaluated rows (%d FAKE, %d REAL), %d excluded.",
        valid_accounting["evaluated_rows"],
        valid_accounting["mapped_ground_truth_counts"]["FAKE"],
        valid_accounting["mapped_ground_truth_counts"]["REAL"],
        valid_accounting["excluded_rows"],
    )

    # 3. Validation inference
    logger.info("Running inference on validation split...")
    valid_statements = [r["statement"] for r in valid_records]
    y_valid = [r["mapped_label"] for r in valid_records]
    probs_real_valid = run_batch_inference_probs(
        valid_statements, model, tokenizer, device, batch_size=batch_size
    )

    # 4. Dense threshold grid search (0.01 to 0.99, step 0.01)
    grid_thresholds = [round(float(th), 2) for th in torch.linspace(0.01, 0.99, 99).tolist()]
    candidate_results: List[Dict[str, Any]] = []

    for th in grid_thresholds:
        preds = [apply_threshold(p, th) for p in probs_real_valid]
        m = compute_threshold_metrics(y_valid, preds)
        candidate_results.append({
            "threshold": th,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "fake_precision": m["fake_class"]["precision"],
            "fake_recall": m["fake_class"]["recall"],
            "fake_f1": m["fake_class"]["f1_score"],
            "real_precision": m["real_class"]["precision"],
            "real_recall": m["real_class"]["recall"],
            "real_f1": m["real_class"]["f1_score"],
            "predicted_counts": m["predicted_counts"],
            "confusion_matrix": m["confusion_matrix"]["matrix"],
        })

    # 5. Predeclared threshold selection rule
    selected_threshold = select_optimal_threshold(
        candidate_results,
        target_metric="macro_f1",
        default_boundary=DEFAULT_DECISION_THRESHOLD,
    )
    logger.info(
        "Selected threshold maximizing validation Macro F1: %.2f",
        selected_threshold,
    )

    # Freeze validation metrics at default (0.50) and selected threshold
    valid_baseline_res = [c for c in candidate_results if math.isclose(c["threshold"], 0.50, abs_tol=1e-4)][0]
    valid_selected_res = [c for c in candidate_results if math.isclose(c["threshold"], selected_threshold, abs_tol=1e-4)][0]

    # 6. Load test split
    test_records, test_accounting = load_and_filter_split(
        "test.tsv", cache_dir=cache_dir
    )
    logger.info(
        "Loaded test split: %d evaluated rows (%d FAKE, %d REAL), %d excluded.",
        test_accounting["evaluated_rows"],
        test_accounting["mapped_ground_truth_counts"]["FAKE"],
        test_accounting["mapped_ground_truth_counts"]["REAL"],
        test_accounting["excluded_rows"],
    )

    # 7. Test inference
    logger.info("Running inference on held-out test split...")
    test_statements = [r["statement"] for r in test_records]
    y_test = [r["mapped_label"] for r in test_records]
    probs_real_test = run_batch_inference_probs(
        test_statements, model, tokenizer, device, batch_size=batch_size
    )

    # 8. Evaluate test at default threshold (0.50)
    preds_test_default = [apply_threshold(p, DEFAULT_DECISION_THRESHOLD) for p in probs_real_test]
    test_baseline_metrics = compute_threshold_metrics(y_test, preds_test_default)

    # Verify baseline reproduction (acc must match 0.4921)
    if not math.isclose(test_baseline_metrics["accuracy"], 0.4921, abs_tol=1e-4):
        raise RuntimeError(
            f"Baseline reproduction mismatch! Expected 0.4921, got {test_baseline_metrics['accuracy']}"
        )

    # 9. Evaluate test at selected threshold
    preds_test_selected = [apply_threshold(p, selected_threshold) for p in probs_real_test]
    test_selected_metrics = compute_threshold_metrics(y_test, preds_test_selected)

    # 10. Assemble compact predictions for auditability (no statement text)
    compact_test_predictions: List[Dict[str, Any]] = []
    for rec, p_real, p_def, p_sel in zip(
        test_records, probs_real_test, preds_test_default, preds_test_selected
    ):
        compact_test_predictions.append({
            "id": rec["id"],
            "raw_label": rec["raw_label"],
            "ground_truth": rec["mapped_label"],
            "prob_real": round(float(p_real), 6),
            "pred_default_0_50": p_def,
            "pred_selected": p_sel,
        })

    # Assemble comprehensive report
    report: Dict[str, Any] = {
        "title": "TruthLens — Threshold Sensitivity Experiment Report",
        "experiment_scope": "Evaluation of decision boundary adjustment on LIAR validation and held-out test splits",
        "experiment_timestamp_utc": timestamp,
        "dataset_provenance": {
            "source_name": DATASET_SOURCE_NAME,
            "archive_url": DATASET_ARCHIVE_URL,
            "archive_sha256": DATASET_ARCHIVE_SHA256,
            "operational_mapping": {
                "FAKE": ["false", "pants-fire", "barely-true"],
                "REAL": ["true", "mostly-true"],
                "EXCLUDED": ["half-true"],
            },
        },
        "split_accounting": {
            "validation": valid_accounting,
            "test": test_accounting,
        },
        "threshold_selection_protocol": {
            "definition": "Predict REAL if P(REAL) >= threshold, else FAKE",
            "search_grid": "0.01 to 0.99 in increments of 0.01 (99 candidate thresholds)",
            "selection_rule": "Maximize validation Macro F1; break ties by proximity to default 0.50, then choose lower threshold",
            "selected_threshold": selected_threshold,
            "default_baseline_threshold": DEFAULT_DECISION_THRESHOLD,
        },
        "validation_results": {
            "default_baseline_metrics": valid_baseline_res,
            "selected_threshold_metrics": valid_selected_res,
            "grid_summary_sample": [
                c for c in candidate_results
                if c["threshold"] in [0.01, 0.05, 0.10, 0.13, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
            ],
            "full_grid_results": candidate_results,
        },
        "held_out_test_results": {
            "baseline_threshold_0_50": test_baseline_metrics,
            "selected_threshold": test_selected_metrics,
            "delta_selected_vs_baseline": {
                "accuracy_delta": round(test_selected_metrics["accuracy"] - test_baseline_metrics["accuracy"], 4),
                "macro_f1_delta": round(test_selected_metrics["macro_f1"] - test_baseline_metrics["macro_f1"], 4),
                "fake_recall_delta": round(test_selected_metrics["fake_class"]["recall"] - test_baseline_metrics["fake_class"]["recall"], 4),
                "real_recall_delta": round(test_selected_metrics["real_class"]["recall"] - test_baseline_metrics["real_class"]["recall"], 4),
            },
        },
        "audit_methodology_disclosure": {
            "candidate_selection_source": "Validation split labels only (valid.tsv, 1,036 items).",
            "selection_integrity": (
                "The candidate threshold of 0.13 was chosen strictly by maximizing validation Macro F1. "
                "The threshold choice was not modified after inspecting test performance."
            ),
            "exploratory_test_examination_disclosure": (
                "During exploratory analysis, the test split was evaluated across multiple thresholds "
                "(0.50, 0.13, 0.10, 0.05, 0.01) to observe test sensitivity. Consequently, the test split "
                "should not be described as strictly untouched. However, this exploration demonstrated that "
                "performance remains consistently constrained across all thresholds (test accuracy 49.11% to 49.51%), "
                "confirming that threshold tuning does not resolve the classifier's failure mode."
            ),
            "exploratory_test_grid_sample": [
                {"threshold": 0.50, "accuracy": 0.4921, "macro_f1": 0.4314, "fake_recall": 0.7482, "real_recall": 0.1826, "confusion_matrix": [[416, 140], [376, 84]]},
                {"threshold": 0.13, "accuracy": 0.4941, "macro_f1": 0.4360, "fake_recall": 0.7446, "real_recall": 0.1913, "confusion_matrix": [[414, 142], [372, 88]]},
                {"threshold": 0.10, "accuracy": 0.4951, "macro_f1": 0.4375, "fake_recall": 0.7446, "real_recall": 0.1935, "confusion_matrix": [[414, 142], [371, 89]]},
                {"threshold": 0.05, "accuracy": 0.4951, "macro_f1": 0.4383, "fake_recall": 0.7428, "real_recall": 0.1957, "confusion_matrix": [[413, 143], [370, 90]]},
                {"threshold": 0.01, "accuracy": 0.4911, "macro_f1": 0.4363, "fake_recall": 0.7338, "real_recall": 0.1978, "confusion_matrix": [[408, 148], [369, 91]]},
            ],
            "baseline_preservation": "The original committed baseline metrics (0.4921 accuracy, 0.4314 Macro F1) remain the official benchmark; no production threshold change was made.",
            "generalization_caveat": "This experiment does not serve as proof of generalization or a causal explanation for model errors.",
        },
        "experiment_findings": [
            f"The candidate threshold selected on the validation split is {selected_threshold:.2f} (Macro F1: {valid_selected_res['macro_f1']:.4f} vs. {valid_baseline_res['macro_f1']:.4f} at 0.50).",
            f"On the test split, threshold {selected_threshold:.2f} yielded Accuracy: {test_selected_metrics['accuracy']:.4f} (baseline: {test_baseline_metrics['accuracy']:.4f}) and Macro F1: {test_selected_metrics['macro_f1']:.4f} (baseline: {test_baseline_metrics['macro_f1']:.4f}).",
            f"REAL recall improved only marginally from {test_baseline_metrics['real_class']['recall']:.4f} to {test_selected_metrics['real_class']['recall']:.4f} (88 correct vs 84 correct out of 460 claims).",
            "Multi-threshold test exploration (0.50 down to 0.01) showed test accuracy fluctuates within a narrow band of 49.11% to 49.51%, confirming severe logit saturation.",
            "Because test behavior was explored at multiple thresholds, this comparison is descriptive and exploratory, not a strictly untouched single-trial generalization test.",
            "Recommendation: Reject adopting threshold adjustment in production. It yields negligible empirical gain (+0.20% test accuracy, +0.46% Macro F1) and does not address the underlying domain shift.",
        ],
        "execution_environment": {
            "device": str(device),
            "batch_size": batch_size,
            "pytorch_version": torch.__version__,
        },
    }


    # Determine output destinations
    if output_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(base_dir, "evaluation_results", "threshold_experiment")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "threshold_report.json")
    pred_path = os.path.join(output_dir, "test_predictions_threshold_compact.json")

    with open(report_path, "w", encoding="utf-8") as f_rep:
        json.dump(report, f_rep, indent=2)

    with open(pred_path, "w", encoding="utf-8") as f_p:
        json.dump(compact_test_predictions, f_p, indent=2)

    logger.info("Saved threshold report to %s", report_path)
    logger.info("Saved compact test predictions to %s", pred_path)
    print_experiment_summary(report)
    return report


def print_experiment_summary(report: Dict[str, Any]) -> None:
    """Print formatted summary table comparing baseline and threshold results."""
    sel_th = report["threshold_selection_protocol"]["selected_threshold"]
    v_base = report["validation_results"]["default_baseline_metrics"]
    v_sel = report["validation_results"]["selected_threshold_metrics"]
    t_base = report["held_out_test_results"]["baseline_threshold_0_50"]
    t_sel = report["held_out_test_results"]["selected_threshold"]

    print("\n" + "=" * 76)
    print("TruthLens — Threshold Sensitivity Experiment Summary")
    print("=" * 76)
    print(f"Dataset Archive:       {report['dataset_provenance']['source_name']}")
    print(f"Archive SHA-256:       {report['dataset_provenance']['archive_sha256'][:16]}...")
    print(f"Predeclared Rule:      Maximize Validation Macro F1 (tie-break: dist to 0.50)")
    print(f"Selected Threshold:    {sel_th:.2f} (Default Baseline: 0.50)")
    print("-" * 76)
    print(f"{'Split':<12} | {'Metric':<14} | {'Default (0.50)':<16} | {'Selected (' + f'{sel_th:.2f}' + ')':<16} | {'Delta':<8}")
    print("-" * 76)
    print(f"{'Validation':<12} | {'Accuracy':<14} | {v_base['accuracy']:<16.4f} | {v_sel['accuracy']:<16.4f} | {v_sel['accuracy'] - v_base['accuracy']:+.4f}")
    print(f"{'Validation':<12} | {'Macro F1':<14} | {v_base['macro_f1']:<16.4f} | {v_sel['macro_f1']:<16.4f} | {v_sel['macro_f1'] - v_base['macro_f1']:+.4f}")
    print(f"{'Validation':<12} | {'FAKE Recall':<14} | {v_base['fake_recall']:<16.4f} | {v_sel['fake_recall']:<16.4f} | {v_sel['fake_recall'] - v_base['fake_recall']:+.4f}")
    print(f"{'Validation':<12} | {'REAL Recall':<14} | {v_base['real_recall']:<16.4f} | {v_sel['real_recall']:<16.4f} | {v_sel['real_recall'] - v_base['real_recall']:+.4f}")
    print("-" * 76)
    print(f"{'Held-Out Test':<12} | {'Accuracy':<14} | {t_base['accuracy']:<16.4f} | {t_sel['accuracy']:<16.4f} | {t_sel['accuracy'] - t_base['accuracy']:+.4f}")
    print(f"{'Held-Out Test':<12} | {'Macro F1':<14} | {t_base['macro_f1']:<16.4f} | {t_sel['macro_f1']:<16.4f} | {t_sel['macro_f1'] - t_base['macro_f1']:+.4f}")
    print(f"{'Held-Out Test':<12} | {'FAKE Recall':<14} | {t_base['fake_class']['recall']:<16.4f} | {t_sel['fake_class']['recall']:<16.4f} | {t_sel['fake_class']['recall'] - t_base['fake_class']['recall']:+.4f}")
    print(f"{'Held-Out Test':<12} | {'REAL Recall':<14} | {t_base['real_class']['recall']:<16.4f} | {t_sel['real_class']['recall']:<16.4f} | {t_sel['real_class']['recall'] - t_base['real_class']['recall']:+.4f}")
    print("-" * 76)
    print("Held-Out Test Confusion Matrices:")
    cm_base = t_base['confusion_matrix']['matrix']
    cm_sel = t_sel['confusion_matrix']['matrix']
    print(f"  Default (0.50):    Actual FAKE -> [{cm_base[0][0]:<4} FAKE, {cm_base[0][1]:<4} REAL]")
    print(f"                     Actual REAL -> [{cm_base[1][0]:<4} FAKE, {cm_base[1][1]:<4} REAL]")
    print(f"  Selected ({sel_th:.2f}):  Actual FAKE -> [{cm_sel[0][0]:<4} FAKE, {cm_sel[0][1]:<4} REAL]")
    print(f"                     Actual REAL -> [{cm_sel[1][0]:<4} FAKE, {cm_sel[1][1]:<4} REAL]")
    print("=" * 76)
    print("FINDING: Threshold adjustment achieves negligible test improvement (+0.20% accuracy).")
    print("         Decision boundary shifting does NOT solve the underlying pathology.")
    print("RECOMMENDATION: Reject threshold change for production.")
    print("=" * 76 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run decision threshold sensitivity experiment for TruthLens classifier."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model repository ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device: auto, mps, cuda, or cpu (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for experiment report and predictions",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="External directory for caching dataset downloads",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_threshold_experiment(
            model_id=args.model_id,
            batch_size=args.batch_size,
            device_pref=args.device,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        logger.error("Threshold experiment failed: %s", exc, exc_info=True)
        sys.exit(1)
