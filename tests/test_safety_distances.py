import math

import numpy as np
import pytest

from av_safety_eval.analysis.safety_distances import (
    compute_center_distances,
    compute_envelope_adjusted_distances,
    summarize_distance_series,
)


def test_compute_center_distances() -> None:
    ego = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    target = np.array([[3.0, 4.0], [4.0, 5.0], [2.0, 0.0]])

    distances = compute_center_distances(ego, target)

    np.testing.assert_allclose(distances, [5.0, 5.0, 0.0])


def test_compute_center_distances_applies_valid_mask() -> None:
    ego = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    target = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    distances = compute_center_distances(ego, target, [True, False, True])

    np.testing.assert_allclose(distances[[0, 2]], [1.0, 3.0])
    assert math.isnan(distances[1])


def test_compute_envelope_adjusted_distances() -> None:
    adjusted = compute_envelope_adjusted_distances(np.array([5.0, 4.0, np.nan]))

    np.testing.assert_allclose(adjusted[:2], [0.5, -0.5])
    assert math.isnan(adjusted[2])


def test_summarize_distance_series_reports_minimum_and_timestep() -> None:
    summary = summarize_distance_series(np.array([4.0, np.nan, 2.5, 3.0]))

    assert summary == {
        "min_distance": 2.5,
        "timestep_of_min": 2,
        "below_near_miss_threshold": True,
        "below_collision_screening_threshold": False,
    }


def test_negative_adjusted_distance_is_overlap_screening() -> None:
    adjusted = compute_envelope_adjusted_distances(np.array([5.0, 4.25]))
    summary = summarize_distance_series(adjusted)

    assert summary["min_distance"] == pytest.approx(-0.25)
    assert summary["timestep_of_min"] == 1
    assert summary["below_collision_screening_threshold"] is True


def test_summarize_all_invalid_series() -> None:
    summary = summarize_distance_series(np.array([np.nan, np.inf]))

    assert math.isnan(summary["min_distance"])
    assert summary["timestep_of_min"] is None
    assert summary["below_near_miss_threshold"] is False
    assert summary["below_collision_screening_threshold"] is False


def test_summarize_ignores_infinity_when_finite_value_exists() -> None:
    summary = summarize_distance_series(np.array([-np.inf, 2.0, np.inf]))

    assert summary["min_distance"] == pytest.approx(2.0)
    assert summary["timestep_of_min"] == 1


def test_distance_helpers_reject_invalid_shapes_and_radii() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        compute_center_distances(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="valid_mask"):
        compute_center_distances(np.zeros((2, 2)), np.zeros((2, 2)), [True])
    with pytest.raises(ValueError, match="ego_radius_m"):
        compute_envelope_adjusted_distances(np.array([1.0]), ego_radius_m=-1.0)
