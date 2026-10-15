"""Tests for Candidate Model Evaluation (Milestone 6D).

Verifies strict protocol enforcement, operational mapping, metric calculation,
confusion matrix orientation, screening criteria checks, and artifact integrity.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure backend directory and repo root are in sys.path
backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
for p in [str(backend_dir), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.evaluate_candidates import (
    ALLOWED_SPLIT_FILENAME,
    DATASET_ARCHIVE_SHA256,
    DATASET_ARCHIVE_URL,
    OPERATIONAL_LABEL_MAP,
    SCREENING_CRITERIA,
    compute_metrics,
    evaluate_screening_criteria,
    load_validation_split,
    map_liar_label,
)


class TestCandidateEvaluationProtocol:
    """Verifies strict adherence to validation-only protocol and data mapping."""

    def test_allowed_split_strictly_enforced(self):
        """Confirm that any attempt to evaluate test.tsv raises ValueError."""
        assert ALLOWED_SPLIT_FILENAME == "valid.tsv"
        with pytest.raises(ValueError, match="Strict Protocol Violation"):
            load_validation_split(Path("fake_path.zip"), split_filename="test.tsv")

    def test_operational_label_mapping(self):
        """Confirm LIAR operational mapping adheres to benchmark specifications."""
        assert map_liar_label("false") == "FAKE"
        assert map_liar_label("pants-fire") == "FAKE"
        assert map_liar_label("barely-true") == "FAKE"
        assert map_liar_label("true") == "REAL"
        assert map_liar_label("mostly-true") == "REAL"
        assert map_liar_label("half-true") is None  # Excluded

    def test_whitespace_and_case_insensitivity(self):
        """Ensure whitespace trimming and case insensitivity."""
        assert map_liar_label("  FALSE  ") == "FAKE"
        assert map_liar_label("Pants-Fire") == "FAKE"
        assert map_liar_label("TRUE") == "REAL"
        assert map_liar_label("Mostly-True") == "REAL"
        assert map_liar_label("Half-True") is None


class TestMetricsAndScreeningCriteria:
    """Verifies metric computation and predeclared screening criteria."""

    def test_perfect_predictions(self):
        """Confirm metrics on 100% accurate predictions."""
        y_true = ["FAKE", "FAKE", "REAL", "REAL"]
        y_pred = ["FAKE", "FAKE", "REAL", "REAL"]
        raw_labels = ["pants-fire", "false", "true", "mostly-true"]

        metrics = compute_metrics(y_true, y_pred, raw_labels=raw_labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["fake_class"]["recall"] == 1.0
        assert metrics["real_class"]["recall"] == 1.0
        assert metrics["confusion_matrix"]["matrix"] == [[2, 0], [0, 2]]
        assert metrics["polar_extreme_accuracy"] == 1.0

        screening = evaluate_screening_criteria(metrics)
        assert screening["all_criteria_met"] is True

    def test_confusion_matrix_orientation(self):
        """Verify confusion matrix orientation is rows actual [FAKE, REAL], cols predicted [FAKE, REAL]."""
        y_true = ["FAKE", "FAKE", "REAL", "REAL"]
        y_pred = ["FAKE", "REAL", "FAKE", "REAL"]
        # Matrix:
        # Actual FAKE: 1 predicted FAKE, 1 predicted REAL
        # Actual REAL: 1 predicted FAKE, 1 predicted REAL
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [1, 1]]

    def test_screening_criteria_failure_on_low_real_recall(self):
        """Confirm screening fails when REAL recall is below 40% even if accuracy is high."""
        # Simulated scenario: 60 FAKE, 40 REAL.
        # Predict all FAKE: Accuracy = 60%, but REAL recall = 0%
        y_true = ["FAKE"] * 60 + ["REAL"] * 40
        y_pred = ["FAKE"] * 100
        raw_labels = ["pants-fire"] * 60 + ["true"] * 40

        metrics = compute_metrics(y_true, y_pred, raw_labels=raw_labels)
        screening = evaluate_screening_criteria(metrics)

        assert screening["all_criteria_met"] is False
        assert screening["detailed_checks"]["real_recall_at_least_40"]["passed"] is False

    def test_screening_criteria_failure_on_low_macro_f1(self):
        """Confirm screening fails when macro F1 is below 0.55."""
        dummy_metrics = {
            "accuracy": 0.62,
            "real_class": {"recall": 0.45},
            "macro_f1": 0.52,  # Below 0.55
            "polar_extreme_accuracy": 0.55,
        }
        screening = evaluate_screening_criteria(dummy_metrics)
        assert screening["all_criteria_met"] is False
        assert screening["detailed_checks"]["macro_f1_at_least_0_55"]["passed"] is False

    def test_length_mismatch_raises_error(self):
        """Confirm mismatched length inputs raise ValueError."""
        with pytest.raises(ValueError, match="identical length"):
            compute_metrics(["FAKE"], ["FAKE", "REAL"])


class TestCandidateArtifactIntegrity:
    """Verifies that generated evaluation artifacts meet all schema and privacy constraints."""

    @pytest.fixture
    def report_json_path(self) -> Path:
        return Path("backend/evaluation_results/candidate_validation/validation_report.json")

    @pytest.fixture
    def report_md_path(self) -> Path:
        return Path("backend/evaluation_results/candidate_validation/validation_report.md")

    def test_json_report_exists_and_valid(self, report_json_path):
        if not report_json_path.exists():
            pytest.skip("validation_report.json has not been generated yet.")

        with open(report_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "protocol" in data
        assert "majority_class_baseline" in data
        assert "models" in data

        proto = data["protocol"]
        assert proto["archive_sha256"] == DATASET_ARCHIVE_SHA256
        assert proto["archive_source_url"] == DATASET_ARCHIVE_URL
        assert proto["accounting"]["split_filename"] == "valid.tsv"
        assert proto["accounting"]["total_rows"] == 1284
        assert proto["accounting"]["excluded_rows"] == 248
        assert proto["accounting"]["evaluated_rows"] == 1036
        assert proto["accounting"]["mapped_ground_truth_counts"] == {"FAKE": 616, "REAL": 420}

        # Check models evaluated
        models = data["models"]
        for role in ["baseline", "candidate_1", "candidate_2"]:
            assert role in models
            m_info = models[role]
            assert "model_id" in m_info
            assert "metrics" in m_info
            assert "screening_outcome" in m_info
            assert "confusion_matrix" in m_info["metrics"]

    def test_markdown_report_exists_and_contains_required_sections(self, report_md_path):
        if not report_md_path.exists():
            pytest.skip("validation_report.md has not been generated yet.")

        content = report_md_path.read_text(encoding="utf-8")
        assert "Evaluation Protocol & Accounting" in content
        assert "Predeclared Screening Criteria" in content
        assert "Comparative Validation Metrics" in content
        assert "Confusion Matrices" in content
        assert "Screening Criteria Evaluation" in content
        assert "Analytical Findings & Structural Conclusions" in content
        assert "test.tsv" not in content.replace("`test.tsv` was not accessed", "")  # Only mentioned to confirm it was not used

    def test_no_raw_statement_text_or_local_paths_in_json(self, report_json_path):
        if not report_json_path.exists():
            pytest.skip("validation_report.json has not been generated yet.")

        raw_text = report_json_path.read_text(encoding="utf-8")
        assert "/Users/" not in raw_text
        assert "statements" not in raw_text
