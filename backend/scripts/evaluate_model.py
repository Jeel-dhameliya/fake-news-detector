#!/usr/bin/env python3
"""TruthLens — Standalone Model Evaluation Script.

Evaluates the frozen classification model (dima806/fake-news-classifier)
against the test split of the LIAR benchmark under a predeclared binary
operational label mapping.

Data Source Provenance:
    The test split was obtained directly from the official UCSB-hosted archive
    (https://www.cs.ucsb.edu/~william/data/liar_dataset.zip), which is the authoritative
    upstream source for the LIAR benchmark. The Hugging Face dataset card `ucsbai/liar`
    (revision commit `ff3a15312273f5294c4418c1c3c28781e1d51afe`) provides only a legacy
    dataset script (`liar.py`) that delegates to this same UCSB archive. Because modern
    Hugging Face datasets library deprecates arbitrary script execution, the UCSB archive
    is retrieved directly and its checksum is verified.

Research Framing & Audit Integrity:
    This evaluation assesses agreement between the model's binary outputs and an
    operational binary mapping of fine-grained political statement veracity ratings.
    It measures out-of-domain agreement only, NOT universal fact-checking accuracy
    or proof of factual truth.
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_model")

# Constant definitions
DEFAULT_MODEL_ID = "dima806/fake-news-classifier"

# Authoritative dataset provenance
DATASET_SOURCE_NAME = "LIAR Benchmark (UCSB original publication archive)"
DATASET_ARCHIVE_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
DATASET_ARCHIVE_SHA256 = "611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7"
DATASET_ARCHIVE_SIZE_BYTES = 1013571
DATASET_RETRIEVAL_TIMESTAMP_UTC = "2026-10-14T07:19:28.888607+00:00"
RELATED_HF_REPO = "ucsbai/liar"
RELATED_HF_COMMIT = "ff3a15312273f5294c4418c1c3c28781e1d51afe"
TEST_FILE_NAME = "test.tsv"
EXPECTED_TEST_ROWS = 1283

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
    """Map a raw LIAR dataset label to the operational binary label.

    Returns:
        'FAKE', 'REAL', or None if excluded / ambiguous / invalid.
    """
    normalized = label.strip().lower()
    return OPERATIONAL_LABEL_MAP.get(normalized, None)


def resolve_device(preferred: str = "auto") -> torch.device:
    """Resolve optimal PyTorch device supporting Apple Silicon MPS, CUDA, and CPU."""
    pref = preferred.strip().lower()
    if pref == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        logger.warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    elif pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    elif pref == "cpu":
        return torch.device("cpu")

    # Auto detection: MPS (Apple Silicon) -> CUDA -> CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_and_extract_liar_test(cache_dir: Optional[str] = None) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Retrieve the official LIAR test split from external cache or download URL.

    Downloaded zip archive is kept in the external user cache directory
    (e.g. ~/.cache/truthlens_eval), never committed to the repository.
    Verifies the archive SHA-256 checksum against the documented benchmark archive.

    Returns:
        Tuple of (list of row dicts containing 'id', 'label', 'statement', provenance_dict).
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/truthlens_eval")
    os.makedirs(cache_dir, exist_ok=True)

    zip_path = os.path.join(cache_dir, "liar_dataset.zip")
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        logger.info("Downloading LIAR dataset from %s to %s...", DATASET_ARCHIVE_URL, zip_path)
        req = urllib.request.Request(
            DATASET_ARCHIVE_URL,
            headers={"User-Agent": "TruthLens-Evaluator/1.0"},
        )
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f_out:
            f_out.write(resp.read())
        logger.info("Download completed successfully (%d bytes).", os.path.getsize(zip_path))
    else:
        logger.info("Using cached LIAR archive: %s", zip_path)

    # Compute SHA-256 checksum of the archive
    with open(zip_path, "rb") as f_archive:
        archive_bytes = f_archive.read()
        computed_sha256 = hashlib.sha256(archive_bytes).hexdigest()
        actual_size = len(archive_bytes)

    if computed_sha256 != DATASET_ARCHIVE_SHA256:
        raise ValueError(
            f"Archive checksum mismatch! Expected SHA-256 {DATASET_ARCHIVE_SHA256}, "
            f"got {computed_sha256}."
        )

    provenance = {
        "source_name": DATASET_SOURCE_NAME,
        "archive_url": DATASET_ARCHIVE_URL,
        "archive_sha256": computed_sha256,
        "archive_size_bytes": actual_size,
        "archive_path_local": zip_path,
        "test_file_name": TEST_FILE_NAME,
    }

    # Read test.tsv from the zip archive
    rows: List[Dict[str, str]] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        if TEST_FILE_NAME not in z.namelist():
            raise FileNotFoundError(f"{TEST_FILE_NAME} not found in downloaded LIAR zip archive.")
        with z.open(TEST_FILE_NAME) as f_tsv:
            reader = csv.reader(
                io.TextIOWrapper(f_tsv, encoding="utf-8"),
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
            )
            for row in reader:
                if len(row) < 3:
                    continue
                rows.append({
                    "id": row[0].strip(),
                    "label": row[1].strip(),
                    "statement": row[2].strip(),
                })

    if len(rows) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Schema mismatch: expected {EXPECTED_TEST_ROWS} test rows, but parsed {len(rows)}."
        )

    return rows, provenance


def compute_evaluation_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics and 2x2 confusion matrix.

    Metrics include:
        - Accuracy
        - Precision, Recall, F1 for FAKE (positive class)
        - Precision, Recall, F1 for REAL
        - Macro F1
        - Weighted F1
        - Clearly labeled 2x2 confusion matrix:
            Convention: rows = actual [FAKE, REAL], columns = predicted [FAKE, REAL]

    Uses zero_division=0 to guarantee finite, well-defined results.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical lengths.")
    if not y_true:
        raise ValueError("Cannot compute metrics on empty ground truth.")

    acc = float(accuracy_score(y_true, y_pred))

    # Binary metrics for FAKE as positive class
    p_fake, r_fake, f1_fake, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label="FAKE", zero_division=0
    )

    # Binary metrics for REAL class
    p_real, r_real, f1_real, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label="REAL", zero_division=0
    )

    # Macro and Weighted F1
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Confusion Matrix: rows = actual [FAKE, REAL], columns = predicted [FAKE, REAL]
    cm = confusion_matrix(y_true, y_pred, labels=["FAKE", "REAL"]).tolist()

    metrics = {
        "accuracy": round(acc, 4),
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
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": {
            "convention": "Rows represent actual classes [FAKE, REAL]; columns represent predicted classes [FAKE, REAL]",
            "row_labels": ["FAKE", "REAL"],
            "col_labels": ["FAKE", "REAL"],
            "matrix": cm,
            "breakdown": {
                "actual_fake_predicted_fake": cm[0][0],  # True Positive (FAKE)
                "actual_fake_predicted_real": cm[0][1],  # False Negative (FAKE)
                "actual_real_predicted_fake": cm[1][0],  # False Positive (FAKE)
                "actual_real_predicted_real": cm[1][1],  # True Negative (FAKE)
            },
        },
    }

    # Verify all metric values are finite and within [0, 1]
    for key, val in [
        ("accuracy", metrics["accuracy"]),
        ("fake_precision", metrics["fake_class"]["precision"]),
        ("fake_recall", metrics["fake_class"]["recall"]),
        ("fake_f1", metrics["fake_class"]["f1_score"]),
        ("real_precision", metrics["real_class"]["precision"]),
        ("real_recall", metrics["real_class"]["recall"]),
        ("real_f1", metrics["real_class"]["f1_score"]),
        ("macro_f1", metrics["macro_f1"]),
        ("weighted_f1", metrics["weighted_f1"]),
    ]:
        if not math.isfinite(val) or not (0.0 <= val <= 1.0):
            raise ValueError(f"Metric {key} is out of valid bounds [0, 1]: {val}")

    return metrics


def run_evaluation(
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 32,
    device_pref: str = "auto",
    output_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute full reproducible evaluation pipeline and generate JSON report."""
    eval_timestamp = datetime.now(timezone.utc).isoformat()
    device = resolve_device(device_pref)
    logger.info("Initializing evaluation with device: %s", device)

    # 1. Load dataset split and verify provenance
    raw_rows, provenance = download_and_extract_liar_test(cache_dir=cache_dir)
    total_test_rows = len(raw_rows)
    logger.info("Successfully loaded %d test split rows from LIAR dataset.", total_test_rows)

    # 2. Count raw labels and apply operational mapping
    raw_label_counts: Dict[str, int] = {}
    evaluated_records: List[Dict[str, str]] = []  # dict with id, raw_label, statement, mapped_label
    excluded_counts: Dict[str, int] = {}

    for row in raw_rows:
        raw_lbl = row["label"]
        raw_label_counts[raw_lbl] = raw_label_counts.get(raw_lbl, 0) + 1
        stmt = row["statement"]

        # Check for empty statements
        if not stmt:
            excluded_counts["empty_statement"] = excluded_counts.get("empty_statement", 0) + 1
            continue

        mapped = map_liar_label(raw_lbl)
        if mapped is None:
            # Excluded (e.g. half-true or unknown)
            excluded_counts[raw_lbl] = excluded_counts.get(raw_lbl, 0) + 1
        else:
            evaluated_records.append({
                "id": row["id"],
                "raw_label": raw_lbl,
                "statement": stmt,
                "mapped_label": mapped,
            })

    evaluated_rows = len(evaluated_records)
    total_excluded = sum(excluded_counts.values())

    if evaluated_rows + total_excluded != total_test_rows:
        raise RuntimeError(
            f"Accounting discrepancy: evaluated ({evaluated_rows}) + excluded ({total_excluded}) "
            f"!= total test rows ({total_test_rows})"
        )

    logger.info(
        "Evaluated rows: %d | Excluded rows: %d | Total test rows: %d",
        evaluated_rows,
        total_excluded,
        total_test_rows,
    )

    # 3. Load model and tokenizer (weights frozen, no fine-tuning)
    logger.info("Loading model and tokenizer for %s...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    # Extract actual id2label from model config
    raw_id2label = getattr(model.config, "id2label", {0: "FAKE", 1: "REAL"})
    id2label: Dict[int, str] = {int(k): str(v).upper() for k, v in raw_id2label.items()}
    logger.info("Verified model id2label mapping: %s", id2label)

    # 4. Perform batched inference in evaluation mode
    statements = [rec["statement"] for rec in evaluated_records]
    y_true = [rec["mapped_label"] for rec in evaluated_records]
    y_pred: List[str] = []

    logger.info("Starting batched inference (batch_size=%d)...", batch_size)
    with torch.inference_mode():
        for i in range(0, len(statements), batch_size):
            batch_texts = statements[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)
            batch_preds = torch.argmax(outputs.logits, dim=-1).tolist()
            y_pred.extend([id2label[p] for p in batch_preds])

    # 5. Compute metrics
    metrics = compute_evaluation_metrics(y_true, y_pred)

    # Count predicted labels
    pred_counts: Dict[str, int] = {}
    for p in y_pred:
        pred_counts[p] = pred_counts.get(p, 0) + 1

    # Attach prediction to evaluated records for compact export
    compact_predictions: List[Dict[str, str]] = []
    for rec, pred in zip(evaluated_records, y_pred):
        compact_predictions.append({
            "id": rec["id"],
            "raw_label": rec["raw_label"],
            "ground_truth": rec["mapped_label"],
            "predicted": pred,
        })

    # Determine output file destinations
    if output_file is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_file = os.path.join(base_dir, "evaluation_results", "liar_test_report.json")
    compact_pred_file = os.path.join(
        os.path.dirname(os.path.abspath(output_file)),
        "liar_predictions_compact.json",
    )

    # 6. Assemble complete reproducible report with strictly audited claims
    report: Dict[str, Any] = {
        "title": "TruthLens — Out-of-Domain Model Evaluation Report",
        "evaluation_scope": "Exploratory agreement assessment under operational binary mapping",
        "evaluation_timestamp_utc": eval_timestamp,
        "model": {
            "model_id": model_id,
            "architecture": model.config.architectures[0] if getattr(model.config, "architectures", None) else "BertForSequenceClassification",
            "id2label": id2label,
            "weights_frozen": True,
        },
        "dataset": {
            "source_name": DATASET_SOURCE_NAME,
            "source_archive_url": DATASET_ARCHIVE_URL,
            "archive_sha256": provenance["archive_sha256"],
            "archive_size_bytes": provenance["archive_size_bytes"],
            "retrieval_timestamp_utc": DATASET_RETRIEVAL_TIMESTAMP_UTC,
            "test_file_name": TEST_FILE_NAME,
            "split": "test",
            "license": "unknown",
            "licensing_limitation_note": (
                "Hugging Face and original LIAR metadata list license as unknown (scraped from PolitiFact API). "
                "Retrieved strictly for academic research and model evaluation. Statement texts are not redistributed."
            ),
            "text_field_evaluated": "statement",
            "metadata_fields_excluded": ["speaker", "party", "context", "state", "job_title", "history"],
            "relationship_to_huggingface": (
                f"The Hugging Face dataset {RELATED_HF_REPO} (commit {RELATED_HF_COMMIT}) contains only a legacy "
                "dataset script (`liar.py`) that delegates to this same UCSB archive URL. Because modern "
                "Hugging Face datasets library deprecates arbitrary script execution, the evaluated data was "
                "downloaded directly from the authoritative UCSB zip archive and verified via SHA-256."
            ),
        },
        "operational_mapping": {
            "protocol": {
                "FAKE": ["false", "pants-fire", "barely-true"],
                "REAL": ["true", "mostly-true"],
                "EXCLUDED": ["half-true"],
            },
            "research_rationale": (
                "The LIAR benchmark provides 6-degree fine-grained truthfulness ratings for political statements, "
                "whereas dima806/fake-news-classifier outputs binary classification into FAKE and REAL classes. "
                "This operational mapping is a simplifying convention for testing out-of-domain transfer agreement. "
                "It does not constitute universal ground truth or definitive factual accuracy."
            ),
        },
        "sample_counts": {
            "total_test_rows": total_test_rows,
            "evaluated_rows": evaluated_rows,
            "total_excluded_rows": total_excluded,
            "raw_test_label_distribution": raw_label_counts,
            "evaluated_ground_truth_distribution": {
                "FAKE": metrics["fake_class"]["support"],
                "REAL": metrics["real_class"]["support"],
            },
            "predicted_distribution": pred_counts,
            "excluded_label_distribution": excluded_counts,
        },
        "metrics": {
            "accuracy": metrics["accuracy"],
            "positive_class_fake": {
                "precision": metrics["fake_class"]["precision"],
                "recall": metrics["fake_class"]["recall"],
                "f1_score": metrics["fake_class"]["f1_score"],
                "support": metrics["fake_class"]["support"],
            },
            "class_real": {
                "precision": metrics["real_class"]["precision"],
                "recall": metrics["real_class"]["recall"],
                "f1_score": metrics["real_class"]["f1_score"],
                "support": metrics["real_class"]["support"],
            },
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "methodological_limitations": [
            "Exploratory agreement assessment: This evaluation measures agreement between model binary outputs and an operational mapping of LIAR's 6-point truthfulness ratings. It does not establish universal fact-checking accuracy or proof of factual truth.",
            "Operational label mapping: Grouping fine-grained PolitiFact ratings into binary FAKE (false, pants-fire, barely-true) and REAL (true, mostly-true) while excluding ambiguous half-true ratings is a simplifying research convention, not ground-truth equivalence.",
            "Observed prediction distribution: The model demonstrated a marked tendency to predict FAKE across the evaluated statements (792 out of 1,016, or 77.95%).",
            "Inability to establish causality or training overlap: Training corpus provenance, training-set overlap with political statements, and causal mechanisms for the observed prediction patterns cannot be established from this evaluation alone, as training logs and datasets for dima806/fake-news-classifier are not independently verified.",
            "Confidence scores: Logits and softmax probabilities reflect classification certainty under the model's internal parameters and priors, not calibrated probabilities of real-world veracity.",
            "Non-author benchmark: This independent evaluation was conducted by TruthLens researchers and is not an official benchmark score from the model author.",
        ],
        "compact_predictions_artifact": "backend/evaluation_results/liar_predictions_compact.json",
        "execution_environment": {
            "device": str(device),
            "batch_size": batch_size,
            "pytorch_version": torch.__version__,
        },
    }

    # 7. Write reports to disk
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(report, f_out, indent=2)

    with open(compact_pred_file, "w", encoding="utf-8") as f_pred:
        json.dump(compact_predictions, f_pred, indent=2)

    logger.info("Evaluation report successfully saved to %s", output_file)
    logger.info("Compact predictions artifact saved to %s", compact_pred_file)
    print_cli_summary(report)
    return report


def print_cli_summary(report: Dict[str, Any]) -> None:
    """Print an informative and structured summary table to console."""
    m = report["metrics"]
    cm = report["confusion_matrix"]["matrix"]
    counts = report["sample_counts"]
    ds = report["dataset"]

    print("\n" + "=" * 70)
    print("TruthLens — Model Evaluation Report (LIAR Test Split)")
    print("=" * 70)
    print(f"Model ID:              {report['model']['model_id']}")
    print(f"Dataset Source:        {ds['source_name']}")
    print(f"Archive URL:           {ds['source_archive_url']}")
    print(f"Archive SHA-256:       {ds['archive_sha256'][:16]}...")
    print(f"Dataset License:       {ds['license']} (academic research evaluation only)")
    print(f"Device:                {report['execution_environment']['device']}")
    print(f"Timestamp (UTC):       {report['evaluation_timestamp_utc']}")
    print("-" * 70)
    print(f"Total Test Rows:       {counts['total_test_rows']}")
    print(f"Evaluated Rows:        {counts['evaluated_rows']}")
    print(f"Excluded Rows:         {counts['total_excluded_rows']} (half-true: {counts['excluded_label_distribution'].get('half-true', 0)})")
    print(f"Raw Label Counts:      {counts['raw_test_label_distribution']}")
    print("-" * 70)
    print(f"Overall Accuracy:      {m['accuracy']:.4f}")
    print(f"FAKE (Pos) Precision:  {m['positive_class_fake']['precision']:.4f}")
    print(f"FAKE (Pos) Recall:     {m['positive_class_fake']['recall']:.4f}")
    print(f"FAKE (Pos) F1 Score:   {m['positive_class_fake']['f1_score']:.4f}")
    print(f"REAL Precision:        {m['class_real']['precision']:.4f}")
    print(f"REAL Recall:           {m['class_real']['recall']:.4f}")
    print(f"REAL F1 Score:         {m['class_real']['f1_score']:.4f}")
    print(f"Macro F1:              {m['macro_f1']:.4f}")
    print(f"Weighted F1:           {m['weighted_f1']:.4f}")
    print("-" * 70)
    print("Confusion Matrix Convention:")
    print("  Rows: Actual classes [FAKE, REAL]")
    print("  Columns: Predicted classes [FAKE, REAL]")
    print(f"                 Pred FAKE     Pred REAL")
    print(f"  Actual FAKE       {cm[0][0]:<10}    {cm[0][1]:<10}  (Total: {cm[0][0] + cm[0][1]})")
    print(f"  Actual REAL       {cm[1][0]:<10}    {cm[1][1]:<10}  (Total: {cm[1][0] + cm[1][1]})")
    print(f"  Total Predicted   {cm[0][0] + cm[1][0]:<10}    {cm[0][1] + cm[1][1]:<10}")
    print("=" * 70)
    print("AUDIT FINDING: Agreement under operational binary mapping on test split.")
    print("               Does not establish training-data overlap or causal explanations.")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate frozen dima806/fake-news-classifier on official LIAR test split."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model repository ID (default: dima806/fake-news-classifier)",
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
        "--output-file",
        type=str,
        default=None,
        help="Path to save output JSON report (default: backend/evaluation_results/liar_test_report.json)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="External directory for caching dataset downloads (default: ~/.cache/truthlens_eval)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_evaluation(
            model_id=args.model_id,
            batch_size=args.batch_size,
            device_pref=args.device,
            output_file=args.output_file,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        logger.error("Evaluation script encountered fatal error: %s", exc, exc_info=True)
        sys.exit(1)
