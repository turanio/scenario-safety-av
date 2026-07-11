"""Template for exporting one QCNet prediction as an av_safety_eval artifact.

This template is intended to be copied into, or adapted from, an external
QCNet checkout after inspecting QCNet's output objects. It deliberately does
not import QCNet in the main thesis repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def _validate_export_inputs(
    positions: np.ndarray,
    probabilities: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions must have shape (num_modes, horizon_steps, 2).")
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        raise ValueError("positions must contain at least one mode and one horizon step.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite.")
    if probabilities.ndim != 1:
        raise ValueError("probabilities must have shape (num_modes,).")
    if probabilities.shape[0] != positions.shape[0]:
        raise ValueError("probabilities length must match positions num_modes.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite.")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass.")
    probabilities = probabilities / total
    return positions, probabilities


def extract_from_qcnet_output(raw_output: Any) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Adapt this function after inspecting QCNet outputs.

    Expected return value:
        scenario_id, target_actor_id, positions, probabilities

    where ``positions`` has shape ``(num_modes, horizon_steps, 2)`` in meters
    and ``probabilities`` has shape ``(num_modes,)``.
    """

    raise NotImplementedError("Adapt this function after inspecting QCNet outputs.")


def save_qcnet_prediction_artifact(
    output_path: str | Path,
    scenario_id: str,
    target_actor_id: str,
    positions: np.ndarray,
    probabilities: np.ndarray,
    dt: float = 0.1,
    coordinate_frame: str = "av2_global",
    source: str = "qcnet_av2_validation_smoke_test",
) -> Path:
    """Validate and save one PredictionSet-compatible QCNet `.npz` artifact."""

    positions, probabilities = _validate_export_inputs(positions, probabilities, dt)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        scenario_id=np.array(scenario_id),
        target_actor_id=np.array(target_actor_id),
        dt=np.array(dt),
        positions=positions,
        probabilities=probabilities,
        coordinate_frame=np.array(coordinate_frame),
        source=np.array(source),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output .npz artifact path.")
    args = parser.parse_args()

    raise SystemExit(
        "This is an export template. Copy/adapt it inside the external QCNet "
        "environment, implement extract_from_qcnet_output(), then call "
        "save_qcnet_prediction_artifact()."
    )


if __name__ == "__main__":
    main()
