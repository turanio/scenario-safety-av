"""Create a fake QCNet-shaped artifact for converter testing only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_fake_prediction(num_modes: int = 6, horizon_steps: int = 60, dt: float = 0.1) -> dict:
    """Return deterministic QCNet-shaped arrays for tests and smoke checks."""

    time = np.arange(1, horizon_steps + 1, dtype=float) * dt
    positions = np.zeros((num_modes, horizon_steps, 2), dtype=float)

    lateral_profiles = [
        np.full(horizon_steps, 3.5),
        np.full(horizon_steps, 2.8),
        np.maximum(3.5 - 0.55 * time, 0.35),
        np.maximum(3.5 - 0.35 * time, 1.2),
        np.full(horizon_steps, -3.5),
        np.maximum(3.5 - 0.25 * time, 1.8),
    ]
    longitudinal_speeds = [0.6, 0.8, 1.0, 0.7, 0.5, 0.4]

    for mode_index in range(num_modes):
        profile_index = mode_index % len(lateral_profiles)
        positions[mode_index, :, 0] = 8.0 - longitudinal_speeds[profile_index] * time
        positions[mode_index, :, 1] = lateral_profiles[profile_index]

    probabilities = np.array([0.35, 0.25, 0.15, 0.1, 0.1, 0.05], dtype=float)
    probabilities = probabilities[:num_modes]
    probabilities = probabilities / probabilities.sum()

    return {
        "scenario_id": np.array("fake_qcnet_converter_test"),
        "target_actor_id": np.array("fake_target"),
        "dt": np.array(dt),
        "positions": positions,
        "probabilities": probabilities,
        "coordinate_frame": np.array("synthetic_global"),
        "source": np.array("fake artifact for converter testing only"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qcnet_smoke/fake_qcnet_prediction.npz"),
        help="Output .npz path.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_fake_prediction()
    np.savez(args.output, **artifact)
    print(f"Fake QCNet artifact for converter testing only written to {args.output}")
    print("modes: 6")
    print("horizon_steps: 60")
    print("dt: 0.1")


if __name__ == "__main__":
    main()
