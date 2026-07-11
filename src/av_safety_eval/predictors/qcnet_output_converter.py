"""Converters for dependency-free QCNet prediction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from av_safety_eval.common.types import PredictionSet, Trajectory


REQUIRED_NPZ_KEYS = {
    "scenario_id",
    "target_actor_id",
    "dt",
    "positions",
    "probabilities",
    "coordinate_frame",
    "source",
}


def _read_scalar_text(value: Any, key: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{key} must be a scalar string.")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    text = str(item)
    if not text:
        raise ValueError(f"{key} must not be empty.")
    return text


def _read_positive_float(value: Any, key: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{key} must be a scalar float.")
    number = float(array.item())
    if not np.isfinite(number):
        raise ValueError(f"{key} must be finite.")
    if number <= 0.0:
        raise ValueError(f"{key} must be positive.")
    return number


def _normalise_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must have shape (num_modes,).")
    if probabilities.size == 0:
        raise ValueError("probabilities must contain at least one mode.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite.")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative.")

    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass.")
    if not np.isclose(total, 1.0, rtol=1e-3, atol=1e-6):
        raise ValueError("probabilities must sum to 1 within tolerance.")
    return probabilities / total


def load_qcnet_npz_prediction(path: str | Path) -> PredictionSet:
    """Load a simple QCNet `.npz` artifact into a ``PredictionSet``.

    The artifact is produced outside this repository by a QCNet/Argoverse 2
    smoke test. This converter intentionally does not import QCNet, AV2,
    PyTorch, or CARLA.
    """

    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"QCNet prediction artifact does not exist: {artifact_path}")

    with np.load(artifact_path) as data:
        missing_keys = sorted(REQUIRED_NPZ_KEYS.difference(data.files))
        if missing_keys:
            raise ValueError(f"QCNet prediction artifact is missing keys: {missing_keys}")

        scenario_id = _read_scalar_text(data["scenario_id"], "scenario_id")
        target_actor_id = _read_scalar_text(data["target_actor_id"], "target_actor_id")
        _read_scalar_text(data["coordinate_frame"], "coordinate_frame")
        _read_scalar_text(data["source"], "source")
        dt = _read_positive_float(data["dt"], "dt")

        positions = np.asarray(data["positions"], dtype=float)
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("positions must have shape (num_modes, horizon_steps, 2).")
        if positions.shape[0] == 0 or positions.shape[1] == 0:
            raise ValueError("positions must contain at least one mode and one horizon step.")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must be finite.")

        probabilities = _normalise_probabilities(np.asarray(data["probabilities"], dtype=float))
        if probabilities.shape[0] != positions.shape[0]:
            raise ValueError("probabilities length must match positions num_modes.")

    trajectories = [
        Trajectory(agent_id=target_actor_id, positions=mode_positions, dt=dt)
        for mode_positions in positions
    ]
    prediction = PredictionSet(
        agent_id=target_actor_id,
        trajectories=trajectories,
        probabilities=probabilities.tolist(),
    )
    prediction.scenario_id = scenario_id  # type: ignore[attr-defined]
    return prediction
