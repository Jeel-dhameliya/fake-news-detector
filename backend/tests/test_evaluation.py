"""Unit tests for model evaluation utilities and metric computations.

Tests operational label mapping and metric calculation using synthetic data
without downloading datasets or invoking remote model endpoints.
"""

import json
import math
import os
import sys
from pathlib import Path
import pytest

# Ensure backend directory and repo root are in sys.path
backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
for p in [str(backend_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from scripts.evaluate_model import (
        compute_evaluation_metrics,
        map_liar_label,
        OPERATIONAL_LABEL_MAP,
    )
except ImportError:
    from backend.scripts.evaluate_model import (
        compute_evaluation_metrics,
        map_liar_label,
        OPERATIONAL_LABEL_MAP,
    )



class TestLabelMapping:
    """Tests for LIAR to binary operational label mapping."""

    def test_fake_mappings(self):
        """Labels corresponding to fake/untrue should map to FAKE."""
        assert map_liar_label("false") == "FAKE"
        assert map_liar_label("pants-fire") == "FAKE"
        assert map_liar_label("barely-true") == "FAKE"

    def test_real_mappings(self):
        """Labels corresponding to truthful claims should map to REAL."""
        assert map_liar_label("true") == "REAL"
        assert map_liar_label("mostly-true") == "REAL"

    def test_ambiguous_half_true_excluded(self):
        """half-true must be excluded as ambiguous under binary protocol."""
        assert map_liar_label("half-true") is None

    def test_whitespace_and_case_insensitivity(self):
        """Mapping should handle whitespace and uppercase variants."""
        assert map_liar_label("  FALSE  ") == "FAKE"
        assert map_liar_label("Pants-Fire") == "FAKE"
        assert map_liar_label("TRUE") == "REAL"
        assert map_liar_label(" Mostly-True ") == "REAL"

    def test_unexpected_and_empty_labels(self):
        """Invalid or unexpected labels should return None."""
        assert map_liar_label("") is None
        assert map_liar_label("unknown_label") is None
        assert map_liar_label("unverified") is None


class TestMetricsComputation:
    """Tests for compute_evaluation_metrics with synthetic ground truth and predictions."""

    def test_perfect_predictions(self):
        """Verify metrics on 100% accurate predictions."""
        y_true = ["FAKE", "FAKE", "REAL", "REAL"]
        y_pred = ["FAKE", "FAKE", "REAL", "REAL"]

        metrics = compute_evaluation_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["fake_class"]["precision"] == 1.0
        assert metrics["fake_class"]["recall"] == 1.0
        assert metrics["fake_class"]["f1_score"] == 1.0
        assert metrics["real_class"]["precision"] == 1.0
        assert metrics["real_class"]["recall"] == 1.0
        assert metrics["real_class"]["f1_score"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0

        cm = metrics["confusion_matrix"]["matrix"]
        assert cm == [[2, 0], [0, 2]]
        assert metrics["confusion_matrix"]["breakdown"]["actual_fake_predicted_fake"] == 2
        assert metrics["confusion_matrix"]["breakdown"]["actual_fake_predicted_real"] == 0
        assert metrics["confusion_matrix"]["breakdown"]["actual_real_predicted_fake"] == 0
        assert metrics["confusion_matrix"]["breakdown"]["actual_real_predicted_real"] == 2

    def test_balanced_errors(self):
        """Verify metrics on known 50% confusion matrix."""
        y_true = ["FAKE", "FAKE", "REAL", "REAL"]
        y_pred = ["FAKE", "REAL", "REAL", "FAKE"]

        metrics = compute_evaluation_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.5
        assert metrics["fake_class"]["precision"] == 0.5
        assert metrics["fake_class"]["recall"] == 0.5
        assert metrics["fake_class"]["f1_score"] == 0.5
        assert metrics["real_class"]["precision"] == 0.5
        assert metrics["real_class"]["recall"] == 0.5
        assert metrics["real_class"]["f1_score"] == 0.5
        assert metrics["macro_f1"] == 0.5

        cm = metrics["confusion_matrix"]["matrix"]
        assert cm == [[1, 1], [1, 1]]

    def test_zero_division_handling(self):
        """When a class is never predicted, precision should default to 0 without error."""
        y_true = ["FAKE", "REAL"]
        y_pred = ["FAKE", "FAKE"]  # REAL never predicted

        metrics = compute_evaluation_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.5
        assert metrics["real_class"]["precision"] == 0.0
        assert metrics["real_class"]["recall"] == 0.0
        assert metrics["real_class"]["f1_score"] == 0.0
        assert math.isfinite(metrics["real_class"]["precision"])

    def test_all_metric_values_finite_and_bounded(self):
        """All computed metrics must be finite numbers in [0, 1]."""
        y_true = ["FAKE"] * 10 + ["REAL"] * 10
        y_pred = ["FAKE"] * 15 + ["REAL"] * 5

        metrics = compute_evaluation_metrics(y_true, y_pred)
        for val in [
            metrics["accuracy"],
            metrics["fake_class"]["precision"],
            metrics["fake_class"]["recall"],
            metrics["fake_class"]["f1_score"],
            metrics["real_class"]["precision"],
            metrics["real_class"]["recall"],
            metrics["real_class"]["f1_score"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
        ]:
            assert math.isfinite(val)
            assert 0.0 <= val <= 1.0

    def test_length_mismatch_raises_error(self):
        """Mismatched y_true and y_pred lengths must raise ValueError."""
        with pytest.raises(ValueError, match="identical lengths"):
            compute_evaluation_metrics(["FAKE"], ["FAKE", "REAL"])

    def test_empty_input_raises_error(self):
        """Empty inputs must raise ValueError."""
        with pytest.raises(ValueError, match="empty ground truth"):
            compute_evaluation_metrics([], [])


class TestSavedEvaluationReport:
    """Verifies the integrity of the saved JSON evaluation report if present."""

    def test_report_schema_and_counts(self):
        """Verify the generated report satisfies all accounting constraints."""
        report_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "evaluation_results", "liar_test_report.json")
        )
        if not os.path.exists(report_path):
            pytest.skip("Evaluation report not yet generated.")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        counts = report["sample_counts"]
        total_test = counts["total_test_rows"]
        evaluated = counts["evaluated_rows"]
        excluded = counts["total_excluded_rows"]

        # Constraint 1: Evaluated + Excluded == Total
        assert evaluated + excluded == total_test
        assert total_test == 1283
        assert evaluated == 1016
        assert excluded == 267

        # Constraint 2: Confusion matrix total equals evaluated count
        cm = report["confusion_matrix"]["matrix"]
        cm_total = sum(sum(row) for row in cm)
        assert cm_total == evaluated

        # Constraint 3: Label distribution sums match
        gt_dist = counts["evaluated_ground_truth_distribution"]
        assert gt_dist["FAKE"] + gt_dist["REAL"] == evaluated
        assert gt_dist["FAKE"] == 556
        assert gt_dist["REAL"] == 460

        # Constraint 4: Metrics bounded in [0, 1]
        m = report["metrics"]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0
        assert 0.0 <= m["weighted_f1"] <= 1.0
        assert 0.0 <= m["positive_class_fake"]["precision"] <= 1.0
        assert 0.0 <= m["positive_class_fake"]["recall"] <= 1.0
        assert 0.0 <= m["positive_class_fake"]["f1_score"] <= 1.0

        # Constraint 5: Known limitations and license note documented
        assert report["dataset"]["license"] == "unknown"
        assert len(report["methodological_limitations"]) >= 5
        limitations_text = " ".join(report["methodological_limitations"])
        assert "training" in limitations_text.lower()
        assert "causality" in limitations_text.lower() or "causal" in limitations_text.lower()
        assert "operational" in limitations_text.lower()

        # Constraint 6: Authoritative data provenance verified
        ds = report["dataset"]
        assert ds["archive_sha256"] == "611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7"
        assert ds["source_archive_url"] == "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
        assert ds["test_file_name"] == "test.tsv"

    def test_compact_predictions_artifact_integrity(self):
        """Verify the compact predictions JSON artifact satisfies privacy and integrity rules."""
        pred_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "evaluation_results", "liar_predictions_compact.json")
        )
        if not os.path.exists(pred_path):
            pytest.skip("Compact predictions artifact not yet generated.")

        with open(pred_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        assert len(predictions) == 1016

        # Ensure no statements are redistributed in the artifact
        for item in predictions:
            assert set(item.keys()) == {"id", "raw_label", "ground_truth", "predicted"}
            assert "statement" not in item
            assert item["ground_truth"] in {"FAKE", "REAL"}
            assert item["predicted"] in {"FAKE", "REAL"}

        # Verify aggregate counts from predictions
        pred_fake_count = sum(1 for p in predictions if p["predicted"] == "FAKE")
        pred_real_count = sum(1 for p in predictions if p["predicted"] == "REAL")
        gt_fake_count = sum(1 for p in predictions if p["ground_truth"] == "FAKE")
        gt_real_count = sum(1 for p in predictions if p["ground_truth"] == "REAL")

        assert pred_fake_count == 792
        assert pred_real_count == 224
        assert gt_fake_count == 556
        assert gt_real_count == 460
