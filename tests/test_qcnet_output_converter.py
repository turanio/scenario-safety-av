from pathlib import Path

import numpy as np
import pytest

from av_safety_eval.predictors.qcnet_output_converter import load_qcnet_npz_prediction


def _artifact_values(**overrides):
    values = {
        "scenario_id": np.array("scenario_001"),
        "target_actor_id": np.array("actor_7"),
        "dt": np.array(0.1),
        "positions": np.zeros((3, 5, 2), dtype=float),
        "probabilities": np.array([0.5, 0.3, 0.2], dtype=float),
        "coordinate_frame": np.array("av2_global"),
        "source": np.array("qcnet_external_smoke_test"),
    }
    values.update(overrides)
    return values


def _write_npz(path: Path, **overrides) -> Path:
    np.savez(path, **_artifact_values(**overrides))
    return path


def test_valid_npz_artifact_converts_to_prediction_set(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / "prediction.npz")

    prediction = load_qcnet_npz_prediction(path)

    assert prediction.agent_id == "actor_7"
    assert len(prediction.trajectories) == 3
    assert prediction.trajectories[0].positions.shape == (5, 2)
    assert prediction.trajectories[0].dt == pytest.approx(0.1)
    assert prediction.probabilities == pytest.approx([0.5, 0.3, 0.2])


def test_probability_normalization_works(tmp_path: Path) -> None:
    path = _write_npz(
        tmp_path / "prediction.npz",
        probabilities=np.array([0.5, 0.3, 0.199999], dtype=float),
    )

    prediction = load_qcnet_npz_prediction(path)

    assert sum(prediction.probabilities or []) == pytest.approx(1.0)


def test_missing_required_key_raises_value_error(tmp_path: Path) -> None:
    values = _artifact_values()
    values.pop("source")
    path = tmp_path / "prediction.npz"
    np.savez(path, **values)

    with pytest.raises(ValueError, match="missing keys"):
        load_qcnet_npz_prediction(path)


def test_invalid_positions_shape_raises_value_error(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / "prediction.npz", positions=np.zeros((3, 5), dtype=float))

    with pytest.raises(ValueError, match="positions must have shape"):
        load_qcnet_npz_prediction(path)


def test_invalid_probabilities_shape_raises_value_error(tmp_path: Path) -> None:
    path = _write_npz(
        tmp_path / "prediction.npz",
        probabilities=np.array([[0.5, 0.3, 0.2]], dtype=float),
    )

    with pytest.raises(ValueError, match="probabilities must have shape"):
        load_qcnet_npz_prediction(path)


def test_negative_probability_raises_value_error(tmp_path: Path) -> None:
    path = _write_npz(
        tmp_path / "prediction.npz",
        probabilities=np.array([0.5, -0.1, 0.6], dtype=float),
    )

    with pytest.raises(ValueError, match="non-negative"):
        load_qcnet_npz_prediction(path)


def test_non_positive_dt_raises_value_error(tmp_path: Path) -> None:
    path = _write_npz(tmp_path / "prediction.npz", dt=np.array(0.0))

    with pytest.raises(ValueError, match="dt must be positive"):
        load_qcnet_npz_prediction(path)
