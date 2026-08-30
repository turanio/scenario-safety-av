from pathlib import Path

import numpy as np
import pytest

from av_safety_eval.experiments.analyze_qcnet_probabilistic_risk import (
    analyze_artifact,
    compute_threshold_retention,
    verify_reproduction,
)


def _write_artifact(path: Path) -> None:
    np.savez(
        path,
        scenario_id=np.array("synthetic-scenario"),
        target_actor_id=np.array("target"),
        positions=np.array(
            [
                [[4.0, 0.0], [0.2, 0.0], [4.0, 0.0]],
                [[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        probabilities=np.array([0.9, 0.1], dtype=np.float32),
        ego_future_positions=np.zeros((3, 2), dtype=np.float32),
        target_future_positions=np.array(
            [[5.0, 0.0], [0.1, 0.0], [4.0, 0.0]], dtype=np.float32
        ),
        ego_future_valid_mask=np.array([True, True, True]),
        target_future_valid_mask=np.array([True, False, True]),
    )


def test_analyze_artifact_uses_joint_valid_mask_and_computes_risk(tmp_path: Path) -> None:
    artifact = tmp_path / "scenario.npz"
    _write_artifact(artifact)

    row = analyze_artifact(artifact, safety_threshold_m=3.0)

    assert row["top1_mode"] == 0
    assert row["worst_case_mode"] == 1
    assert row["top1_min_distance"] == pytest.approx(4.0)
    assert row["worst_case_min_distance"] == pytest.approx(1.0)
    assert row["ground_truth_min_distance"] == pytest.approx(4.0)
    assert row["unsafe_mode_count"] == 1
    assert row["unsafe_probability_mass"] == pytest.approx(0.1)
    assert row["severity_weighted_risk"] == pytest.approx(0.2)
    assert row["top1_event"] is False
    assert row["worst_case_event"] is True


def test_threshold_retention_uses_top1_fallback(tmp_path: Path) -> None:
    artifact = tmp_path / "scenario.npz"
    _write_artifact(artifact)
    row = analyze_artifact(artifact)

    results = compute_threshold_retention([row], probability_thresholds=[0.0, 0.2, 0.95])

    assert results[0]["triggered_scenarios"] == 1
    assert results[0]["total_unsafe_probability_mass_retained"] == pytest.approx(0.1)
    assert results[0]["total_probability_weighted_severity_retained"] == pytest.approx(0.2)
    assert results[1]["triggered_scenarios"] == 0
    assert results[1]["fallback_scenarios"] == 0
    assert results[1]["mean_retained_probability_mass"] == pytest.approx(0.9)
    assert results[2]["triggered_scenarios"] == 0
    assert results[2]["fallback_scenarios"] == 1
    assert results[2]["mean_eligible_modes"] == pytest.approx(1.0)
    assert results[2]["mean_retained_probability_mass"] == pytest.approx(0.9)


def test_analyze_artifact_rejects_missing_joint_future(tmp_path: Path) -> None:
    artifact = tmp_path / "scenario.npz"
    _write_artifact(artifact)
    with np.load(artifact) as original:
        values = {key: original[key] for key in original.files}
    values["target_future_valid_mask"] = np.zeros(3, dtype=bool)
    np.savez(artifact, **values)

    with pytest.raises(ValueError, match="no jointly valid future timestep"):
        analyze_artifact(artifact)


def test_reproduction_gate_rejects_incomplete_batch(tmp_path: Path) -> None:
    artifact = tmp_path / "scenario.npz"
    _write_artifact(artifact)
    row = analyze_artifact(artifact)
    thresholds = compute_threshold_retention([row])

    with pytest.raises(RuntimeError, match="Reproduction count mismatch"):
        verify_reproduction([row], thresholds, ["synthetic-scenario"])
