import numpy as np
import pytest

from av_safety_eval.experiments.analyze_qcnet_risk_aware_decisions import (
    binary_auprc,
    binary_auroc,
    build_decision_thresholds,
    classification_metrics,
    expected_distance_deficit_risk,
    probability_filter_decisions,
    reliability_rows,
)


def test_expected_distance_deficit_risk_is_normalized() -> None:
    risk = expected_distance_deficit_risk(
        probabilities=[0.6, 0.3, 0.1],
        mode_min_distances=[4.0, 1.5, 0.0],
        safety_threshold_m=3.0,
    )

    assert risk == pytest.approx(0.25)


def test_probability_filter_uses_top1_fallback() -> None:
    records = [
        {
            "probabilities": np.asarray([0.8, 0.2]),
            "mode_min_distances": np.asarray([4.0, 1.0]),
            "top1_mode": 0,
        }
    ]

    assert probability_filter_decisions(records, 0.1).tolist() == [True]
    assert probability_filter_decisions(records, 0.5).tolist() == [False]
    assert probability_filter_decisions(records, 0.95).tolist() == [False]


def test_classification_metrics_reports_realized_outcomes() -> None:
    metrics = classification_metrics(
        decisions=[True, True, False, False],
        realized_events=[True, False, True, False],
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["intervention_rate"] == pytest.approx(0.5)
    assert metrics["realized_event_recall"] == pytest.approx(0.5)
    assert metrics["realized_event_precision"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_threshold_grid_contains_observed_transitions_and_above_maximum() -> None:
    scores = np.asarray([0.0, 0.02404508, 0.141277462])

    thresholds = build_decision_thresholds(scores)

    assert 0.05 in thresholds
    assert 0.02404508 in thresholds
    assert 0.141277462 in thresholds
    assert thresholds[-1] > scores.max()


def test_binary_score_metrics_handle_ties() -> None:
    labels = [False, True, False, True]
    scores = [0.1, 0.9, 0.2, 0.8]

    assert binary_auroc(labels, scores) == pytest.approx(1.0)
    assert binary_auprc(labels, scores) == pytest.approx(1.0)
    assert binary_auroc([False, True], [0.5, 0.5]) == pytest.approx(0.5)
    assert binary_auprc([False, True], [0.5, 0.5]) == pytest.approx(0.5)


def test_reliability_bins_preserve_total_count() -> None:
    rows = reliability_rows(
        labels=[False, True, True],
        scores=[0.0, 0.2, 1.0],
        bin_edges=[0.0, 0.5, 1.0],
    )

    assert sum(int(row["scenario_count"]) for row in rows) == 3
    assert rows[0]["observed_event_rate"] == pytest.approx(0.5)
    assert rows[1]["observed_event_rate"] == pytest.approx(1.0)
