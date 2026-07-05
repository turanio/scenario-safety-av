"""Run the synthetic Constant Velocity baseline experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from av_safety_eval.experiments.baseline_common import (
    evaluate_constant_velocity_baseline,
    project_root,
)
from av_safety_eval.scenarios.synthetic_interaction import SyntheticScenarioConfig


BASELINE_CONFIG = SyntheticScenarioConfig(
    name="synthetic_lane_change",
    ego_x=0.0,
    ego_y=0.0,
    ego_vx=10.0,
    ego_vy=0.0,
    target_x=18.0,
    target_y=3.5,
    target_vx=7.0,
    target_vy=-0.35,
    dt=0.2,
    horizon_steps=30,
)


def run_baseline(
    output_root: str | Path | None = None,
    make_plot: bool = True,
) -> dict[str, Any]:
    """Run a deterministic synthetic baseline and save metrics."""

    root = Path(output_root) if output_root is not None else project_root() / "results"
    return evaluate_constant_velocity_baseline(
        BASELINE_CONFIG,
        output_root=root,
        make_plot=make_plot,
        experiment_name="baseline_constant_velocity_synthetic",
    )


def main() -> None:
    """CLI entry point for the baseline demo."""

    result = run_baseline()
    print("Baseline experiment complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
