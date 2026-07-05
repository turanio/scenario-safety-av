"""Trajectory plotting utilities."""

from __future__ import annotations

import os
from pathlib import Path

_MPLCONFIGDIR = Path(os.environ.get("AV_SAFETY_EVAL_MPLCONFIGDIR", "/tmp/av_safety_eval_matplotlib"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from av_safety_eval.common.types import Trajectory


def _positions(trajectory: Trajectory | np.ndarray) -> np.ndarray:
    if isinstance(trajectory, Trajectory):
        return trajectory.positions
    return np.asarray(trajectory, dtype=float)


def plot_trajectories(
    ego_trajectory: Trajectory | np.ndarray,
    other_trajectory: Trajectory | np.ndarray,
    output_path: str | Path,
    title: str = "Baseline Synthetic Trajectories",
) -> Path:
    """Plot ego and target trajectories to a PNG file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ego_positions = _positions(ego_trajectory)
    other_positions = _positions(other_trajectory)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ego_positions[:, 0], ego_positions[:, 1], label="ego", linewidth=2)
    ax.plot(other_positions[:, 0], other_positions[:, 1], label="target", linewidth=2)
    ax.scatter(ego_positions[0, 0], ego_positions[0, 1], marker="o", label="ego start")
    ax.scatter(other_positions[0, 0], other_positions[0, 1], marker="x", label="target start")
    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
