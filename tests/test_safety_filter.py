import numpy as np
import pytest

from av_safety_eval.planning.safety_filter import (
    BRAKE,
    NO_BRAKE,
    evaluate_probability_aware_filter,
    evaluate_top1_filter,
    evaluate_worst_case_filter,
)


def test_top1_filter_checks_only_highest_probability_mode() -> None:
    result = evaluate_top1_filter(
        per_mode_distances=np.array([4.0, 1.0]),
        mode_probabilities=np.array([0.8, 0.2]),
    )

    assert result.policy_name == "top1"
    assert result.action == NO_BRAKE
    assert result.is_safe is True
    assert result.trigger_mode is None
    assert result.trigger_probability is None
    assert result.min_distance == pytest.approx(4.0)
    assert "Top-1 mode 0" in result.reason


def test_top1_filter_returns_trigger_evidence() -> None:
    result = evaluate_top1_filter(
        per_mode_distances=np.array([2.5, 1.0]),
        mode_probabilities=np.array([0.8, 0.2]),
    )

    assert result.action == BRAKE
    assert result.is_safe is False
    assert result.trigger_mode == 0
    assert result.trigger_probability == pytest.approx(0.8)
    assert result.threshold_m == pytest.approx(3.0)
    assert "below" in result.reason


def test_worst_case_filter_can_trigger_on_lower_probability_mode() -> None:
    result = evaluate_worst_case_filter(
        per_mode_distances=np.array([[4.0, 3.5], [2.5, 1.0]]),
        mode_probabilities=np.array([0.9, 0.1]),
    )

    assert result.policy_name == "worst_case"
    assert result.action == BRAKE
    assert result.trigger_mode == 1
    assert result.trigger_probability == pytest.approx(0.1)
    assert result.min_distance == pytest.approx(1.0)
    assert "closest of all modes" in result.reason


def test_probability_aware_filter_ignores_modes_below_threshold() -> None:
    result = evaluate_probability_aware_filter(
        per_mode_distances=np.array([4.0, 1.0]),
        mode_probabilities=np.array([0.96, 0.04]),
    )

    assert result.policy_name == "probability_aware_p005"
    assert result.action == NO_BRAKE
    assert result.trigger_mode is None
    assert result.min_distance == pytest.approx(4.0)
    assert "1 eligible mode with p >= 0.050000" in result.reason


def test_probability_aware_filter_uses_risky_mode_at_lower_threshold() -> None:
    result = evaluate_probability_aware_filter(
        per_mode_distances=np.array([4.0, 1.0]),
        mode_probabilities=np.array([0.96, 0.04]),
        probability_threshold=0.01,
    )

    assert result.policy_name == "probability_aware_p001"
    assert result.action == BRAKE
    assert result.trigger_mode == 1
    assert result.trigger_probability == pytest.approx(0.04)
    assert "2 modes with p >= 0.010000" in result.reason


def test_probability_aware_filter_includes_mode_at_threshold() -> None:
    result = evaluate_probability_aware_filter(
        per_mode_distances=np.array([4.0, 2.0]),
        mode_probabilities=np.array([0.95, 0.05]),
    )

    assert result.action == BRAKE
    assert result.trigger_mode == 1
    assert result.trigger_probability == pytest.approx(0.05)


def test_probability_aware_filter_falls_back_to_top1() -> None:
    result = evaluate_probability_aware_filter(
        per_mode_distances=np.array([2.0, 1.0]),
        mode_probabilities=np.array([0.6, 0.4]),
        probability_threshold=0.9,
    )

    assert result.action == BRAKE
    assert result.trigger_mode == 0
    assert result.trigger_probability == pytest.approx(0.6)
    assert result.min_distance == pytest.approx(2.0)
    assert "No mode meets p >= 0.900000" in result.reason
    assert "falling back to top-1 mode 0" in result.reason


def test_filter_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        evaluate_top1_filter(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_worst_case_filter(np.array([1.0]), np.array([1.1]))
    with pytest.raises(ValueError, match="at most 1.0"):
        evaluate_probability_aware_filter(
            np.array([1.0]), np.array([1.0]), probability_threshold=1.1
        )
