from pathlib import Path

import numpy as np
import pytest

from av_safety_eval.experiments.validate_qcnet_artifact_cohort import (
    historical_overlap,
    inspect_artifact_structure,
)


def _write_valid_artifact(path: Path) -> None:
    np.savez(
        path,
        scenario_id=np.array(path.stem),
        target_actor_id=np.array("target"),
        positions=np.zeros((6, 60, 2), dtype=np.float32),
        probabilities=np.full(6, 1 / 6, dtype=np.float32),
        ego_future_positions=np.zeros((60, 2), dtype=np.float32),
        target_future_positions=np.ones((60, 2), dtype=np.float32),
        ego_future_valid_mask=np.ones(60, dtype=bool),
        target_future_valid_mask=np.ones(60, dtype=bool),
    )


def test_historical_overlap_reports_shared_ids() -> None:
    assert historical_overlap(["a", "b", "c"], ["c", "d"]) == {"c"}


def test_artifact_structure_accepts_six_mode_sixty_step_schema(tmp_path: Path) -> None:
    artifact = tmp_path / "scenario-a.npz"
    _write_valid_artifact(artifact)

    row = inspect_artifact_structure(artifact)

    assert row["status"] == "pass"
    assert row["num_modes"] == 6
    assert row["prediction_steps"] == 60
    assert row["joint_valid_steps"] == 60


def test_artifact_structure_rejects_wrong_mode_count(tmp_path: Path) -> None:
    artifact = tmp_path / "scenario-a.npz"
    _write_valid_artifact(artifact)
    with np.load(artifact) as original:
        values = {key: original[key] for key in original.files}
    values["positions"] = values["positions"][:5]
    values["probabilities"] = np.full(5, 0.2, dtype=np.float32)
    np.savez(artifact, **values)

    row = inspect_artifact_structure(artifact)

    assert row["status"] == "fail"
    assert "positions shape" in row["notes"]
