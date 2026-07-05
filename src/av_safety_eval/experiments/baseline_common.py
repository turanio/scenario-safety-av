"""Shared helpers for deterministic baseline experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from av_safety_eval.metrics.safety import (
    compute_min_distance,
    compute_time_to_collision,
    is_collision,
    is_near_miss,
)
from av_safety_eval.predictors.constant_velocity import ConstantVelocityPredictor
from av_safety_eval.scenarios.synthetic_interaction import (
    SyntheticInteractionScenario,
    SyntheticScenarioConfig,
)
from av_safety_eval.visualization.trajectory_plot import plot_trajectories

BASELINE_SUMMARY_COLUMNS = [
    "experiment_name",
    "scenario",
    "predictor",
    "horizon_steps",
    "dt",
    "min_distance",
    "time_to_collision",
    "near_miss",
    "collision",
]


def project_root() -> Path:
    """Return the repository root for the current src-layout package."""

    return Path(__file__).resolve().parents[3]


def evaluate_constant_velocity_baseline(
    config: SyntheticScenarioConfig,
    output_root: str | Path | None = None,
    make_plot: bool = False,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate one synthetic scenario with the Constant Velocity baseline."""

    root = Path(output_root) if output_root is not None else project_root() / "results"
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    resolved_experiment_name = experiment_name or f"baseline_constant_velocity_{config.name}"
    scenario = SyntheticInteractionScenario(config)
    state = scenario.reset()
    target = state.agents[0]

    predictor = ConstantVelocityPredictor()
    ego_prediction = predictor.predict(
        [state.ego],
        horizon_steps=config.horizon_steps,
        dt=config.dt,
    )
    target_prediction = predictor.predict(
        [target],
        horizon_steps=config.horizon_steps,
        dt=config.dt,
    )
    ego_trajectory = ego_prediction.trajectories[0]
    target_trajectory = target_prediction.trajectories[0]

    collision_threshold = 1.0
    near_miss_threshold = 3.0
    min_distance = compute_min_distance(ego_trajectory, target_trajectory)
    time_to_collision = compute_time_to_collision(
        state.ego,
        target,
        collision_distance=collision_threshold,
        max_time=config.horizon_steps * config.dt,
    )

    figure_file = None
    if make_plot:
        figure_file = plot_trajectories(
            ego_trajectory,
            target_trajectory,
            figures_dir / f"{resolved_experiment_name}.png",
            title=f"{config.name} Constant Velocity Baseline",
        )

    metrics_file = metrics_dir / f"{resolved_experiment_name}.json"
    result: dict[str, Any] = {
        "experiment_name": resolved_experiment_name,
        "scenario": config.name,
        "predictor": "constant_velocity",
        "horizon_steps": config.horizon_steps,
        "dt": config.dt,
        "min_distance": round(min_distance, 6),
        "time_to_collision": None if math.isinf(time_to_collision) else round(time_to_collision, 6),
        "near_miss": is_near_miss(min_distance, threshold=near_miss_threshold),
        "collision": is_collision(min_distance, threshold=collision_threshold),
    }
    json_result = {
        **result,
        "metrics_file": str(metrics_file),
        "figure_file": str(figure_file) if figure_file else None,
    }
    metrics_file.write_text(json.dumps(json_result, indent=2), encoding="utf-8")
    return json_result


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    """Return the stable CSV row subset for a baseline result."""

    return {column: result[column] for column in BASELINE_SUMMARY_COLUMNS}
