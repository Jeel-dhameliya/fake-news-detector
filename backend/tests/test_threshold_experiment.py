"""Unit tests for decision threshold sensitivity experiment utilities and report integrity.

Tests threshold comparison boundaries, selection tie-breaking rules, metric orientation,
label mapping, and reproducibility without downloading remote datasets during testing.
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

from scripts.threshold_experiment import (
    apply_threshold,
    select_optimal_threshold,
    compute_threshold_metrics,
    map_liar_label,
    DEFAULT_DECISION_THRESHOLD,
    DATASET_ARCHIVE_SHA256,
)


class TestThresholdComparisonBoundary:
    """Verifies precision and behavior of the threshold decision boundary."""

    def test_exact_threshold_boundary_predicts_real(self):
        """When prob_real == threshold, decision rule must evaluate to REAL (>= boundary)."""
        assert apply_threshold(0.50, 0.50) == "REAL"
        assert apply_threshold(0.13, 0.13) == "REAL"
        assert apply_threshold(0.0, 0.0) == "REAL"
        assert apply_threshold(1.0, 1.0) == "REAL"

    def test_strictly_below_threshold_predicts_fake(self):
        """When prob_real < threshold, decision rule must evaluate to FAKE."""
        assert apply_threshold(0.49999, 0.50) == "FAKE"
        assert apply_threshold(0.12999, 0.13) == "FAKE"
        assert apply_threshold(0.0001, 0.05) == "FAKE"

    def test_strictly_above_threshold_predicts_real(self):
        """When prob_real > threshold, decision rule must evaluate to REAL."""
        assert apply_threshold(0.50001, 0.50) == "REAL"
        assert apply_threshold(0.13001, 0.13) == "REAL"
        assert apply_threshold(0.99, 0.95) == "REAL"

    def test_invalid_threshold_raises_value_error(self):
        """Thresholds outside [0.0, 1.0] or non-finite values must raise ValueError."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            apply_threshold(0.5, -0.01)
        with pytest.raises(ValueError, match="Threshold must be in"):
            apply_threshold(0.5, 1.01)
        with pytest.raises(ValueError, match="prob_real must be a finite float"):
            apply_threshold(float("nan"), 0.5)


class TestThresholdSelectionAndTieBreaking:
    """Tests the predeclared threshold selection and deterministic tie-breaking logic."""

    def test_selects_highest_macro_f1(self):
        """Selects candidate threshold that strictly maximizes validation macro F1."""
        candidates = [
            {"threshold": 0.40, "macro_f1": 0.450},
            {"threshold": 0.30, "macro_f1": 0.465},
            {"threshold": 0.50, "macro_f1": 0.460},
        ]
        assert select_optimal_threshold(candidates) == 0.30

    def test_tie_breaking_proximity_to_default_boundary(self):
        """When multiple thresholds tie on macro F1, choose the one closest to 0.50."""
        candidates = [
            {"threshold": 0.10, "macro_f1": 0.468},  # distance to 0.50 = 0.40
            {"threshold": 0.40, "macro_f1": 0.468},  # distance to 0.50 = 0.10 (closest)
            {"threshold": 0.80, "macro_f1": 0.468},  # distance to 0.50 = 0.30
        ]
        assert select_optimal_threshold(candidates) == 0.40

    def test_tie_breaking_lower_threshold_on_symmetric_distance(self):
        """When multiple thresholds tie on macro F1 and distance to 0.50, choose the lower threshold."""
        candidates = [
            {"threshold": 0.40, "macro_f1": 0.470},  # dist = 0.10
            {"threshold": 0.60, "macro_f1": 0.470},  # dist = 0.10
        ]
        # Distance is symmetrically 0.10; lower threshold (0.40) must be chosen deterministically
        assert select_optimal_threshold(candidates) == 0.40

    def test_empty_candidates_raises_error(self):
        """Empty candidate list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            select_optimal_threshold([])


class TestMetricAndConfusionMatrixOrientation:
    """Verifies that compute_threshold_metrics produces the correct confusion matrix orientation."""

    def test_confusion_matrix_orientation(self):
        """Convention: rows actual [FAKE, REAL], columns predicted [FAKE, REAL]."""
        y_true = ["FAKE", "FAKE", "REAL", "REAL"]
        y_pred = ["FAKE", "REAL", "FAKE", "REAL"]

        m = compute_threshold_metrics(y_true, y_pred)
        cm = m["confusion_matrix"]["matrix"]
        # Expected:
        # Row 0 (Actual FAKE): 1 pred FAKE (TP), 1 pred REAL (FN)
        # Row 1 (Actual REAL): 1 pred FAKE (FP), 1 pred REAL (TN)
        assert cm == [[1, 1], [1, 1]]
        assert m["confusion_matrix"]["breakdown"]["actual_fake_predicted_fake"] == 1
        assert m["confusion_matrix"]["breakdown"]["actual_fake_predicted_real"] == 1
        assert m["confusion_matrix"]["breakdown"]["actual_real_predicted_fake"] == 1
        assert m["confusion_matrix"]["breakdown"]["actual_real_predicted_real"] == 1
        assert m["accuracy"] == 0.5
        assert m["balanced_accuracy"] == 0.5
        assert m["macro_f1"] == 0.5

    def test_all_metrics_bounded_and_finite(self):
        """All computed metrics must be finite floats within [0.0, 1.0]."""
        y_true = ["FAKE"] * 20 + ["REAL"] * 20
        y_pred = ["FAKE"] * 25 + ["REAL"] * 15

        m = compute_threshold_metrics(y_true, y_pred)
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            assert math.isfinite(m[key])
            assert 0.0 <= m[key] <= 1.0


class TestLabelMappingProtocol:
    """Verifies operational label mapping and exclusion of half-true."""

    def test_exclusion_of_half_true(self):
        """half-true must be excluded under the binary evaluation protocol."""
        assert map_liar_label("half-true") is None
        assert map_liar_label("Half-True") is None

    def test_fake_mappings(self):
        """false, pants-fire, barely-true map to FAKE."""
        assert map_liar_label("false") == "FAKE"
        assert map_liar_label("pants-fire") == "FAKE"
        assert map_liar_label("barely-true") == "FAKE"

    def test_real_mappings(self):
        """true and mostly-true map to REAL."""
        assert map_liar_label("true") == "REAL"
        assert map_liar_label("mostly-true") == "REAL"


class TestThresholdReportAndArtifactIntegrity:
    """Verifies reproducibility, structure, and accounting of the generated JSON artifacts."""

    def test_threshold_report_structure_and_baseline_reproduction(self):
        """Verify the threshold report reproduces baseline metrics and satisfies accounting constraints."""
        report_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "evaluation_results",
                "threshold_experiment",
                "threshold_report.json",
            )
        )
        if not os.path.exists(report_path):
            pytest.skip("Threshold experiment report not yet generated.")

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # 1. Dataset archive verification
        ds = report["dataset_provenance"]
        assert ds["archive_sha256"] == DATASET_ARCHIVE_SHA256
        assert "liar_dataset.zip" in ds["archive_url"]

        # 2. Split accounting
        val_acc = report["split_accounting"]["validation"]
        assert val_acc["total_rows"] == 1284
        assert val_acc["evaluated_rows"] == 1036
        assert val_acc["excluded_rows"] == 248
        assert val_acc["evaluated_rows"] + val_acc["excluded_rows"] == val_acc["total_rows"]

        test_acc = report["split_accounting"]["test"]
        assert test_acc["total_rows"] == 1283
        assert test_acc["evaluated_rows"] == 1016
        assert test_acc["excluded_rows"] == 267
        assert test_acc["evaluated_rows"] + test_acc["excluded_rows"] == test_acc["total_rows"]

        # 3. Selected threshold
        proto = report["threshold_selection_protocol"]
        assert proto["selected_threshold"] == 0.13
        assert proto["default_baseline_threshold"] == 0.50

        # 4. Baseline reproduction on test split
        test_base = report["held_out_test_results"]["baseline_threshold_0_50"]
        assert math.isclose(test_base["accuracy"], 0.4921, abs_tol=1e-4)
        assert math.isclose(test_base["macro_f1"], 0.4314, abs_tol=1e-4)
        assert test_base["confusion_matrix"]["matrix"] == [[416, 140], [376, 84]]

        # 5. Selected threshold evaluation on test split
        test_sel = report["held_out_test_results"]["selected_threshold"]
        assert math.isclose(test_sel["accuracy"], 0.4941, abs_tol=1e-4)
        assert math.isclose(test_sel["macro_f1"], 0.4360, abs_tol=1e-4)
        assert test_sel["confusion_matrix"]["matrix"] == [[414, 142], [372, 88]]

        # 6. Audit methodology disclosure verification
        audit = report["audit_methodology_disclosure"]
        assert "validation" in audit["candidate_selection_source"].lower()
        assert len(audit["exploratory_test_grid_sample"]) == 5
        assert "causal" in audit["generalization_caveat"].lower()

    def test_compact_test_predictions_artifact_integrity(self):
        """Verify the compact test predictions artifact contains no statement text and valid records."""
        pred_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "evaluation_results",
                "threshold_experiment",
                "test_predictions_threshold_compact.json",
            )
        )
        if not os.path.exists(pred_path):
            pytest.skip("Compact test predictions artifact not yet generated.")

        with open(pred_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        assert len(predictions) == 1016

        for item in predictions:
            assert set(item.keys()) == {
                "id",
                "raw_label",
                "ground_truth",
                "prob_real",
                "pred_default_0_50",
                "pred_selected",
            }
            assert "statement" not in item
            assert "text" not in item
            assert 0.0 <= item["prob_real"] <= 1.0
            assert item["ground_truth"] in {"FAKE", "REAL"}
            assert item["pred_default_0_50"] in {"FAKE", "REAL"}
            assert item["pred_selected"] in {"FAKE", "REAL"}
