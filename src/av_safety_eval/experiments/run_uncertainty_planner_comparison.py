"""Compare deterministic and synthetic uncertainty-aware planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from av_safety_eval.common.types import AgentState, PredictionSet
from av_safety_eval.experiments.baseline_common import project_root
from av_safety_eval.metrics.safety import compute_time_to_collision, is_collision, is_near_miss
from av_safety_eval.planners.base import Planner
from av_safety_eval.planners.conservative_uncertainty_planner import ConservativeUncertaintyPlanner
from av_safety_eval.planners.standard_planner import StandardPlanner
from av_safety_eval.predictors.base import TrajectoryPredictor
from av_safety_eval.predictors.constant_velocity import ConstantVelocityPredictor
from av_safety_eval.predictors.synthetic_multimodal import SyntheticMultimodalPredictor
from av_safety_eval.scenarios.base import Scenario
from av_safety_eval.scenarios.delayed_cut_in import DelayedCutInScenario
from av_safety_eval.scenarios.synthetic_interaction import (
    SyntheticInteractionScenario,
    SyntheticScenarioConfig,
    ambiguous_cut_in_config,
    delayed_cut_in_config,
)

UNCERTAINTY_SUMMARY_COLUMNS = [
    "scenario",
    "planner",
    "predictor",
    "prediction_modes",
    "steps",
    "final_time",
    "min_distance",
    "near_miss",
    "collision",
    "intervention_count",
    "success",
]

UNCERTAINTY_LOG_COLUMNS = [
    "step",
    "time",
    "ego_x",
    "ego_y",
    "ego_vx",
    "ego_vy",
    "target_x",
    "target_y",
    "target_vx",
    "target_vy",
    "action_acceleration",
    "action_steering",
    "min_distance",
    "time_to_collision",
    "near_miss",
    "collision",
]

COLLISION_THRESHOLD = 1.0
NEAR_MISS_THRESHOLD = 3.0


def _instant_distance(ego: AgentState, target: AgentState) -> float:
    return float(np.linalg.norm(ego.position - target.position))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def uncertainty_scenario_configs() -> list[SyntheticScenarioConfig]:
    """Return scenarios used for synthetic uncertainty-aware evaluation."""

    return [ambiguous_cut_in_config(), delayed_cut_in_config()]


def build_uncertainty_scenario(config: SyntheticScenarioConfig) -> Scenario:
    """Build the scenario implementation for a synthetic uncertainty config."""

    if config.name == "delayed_cut_in":
        return DelayedCutInScenario(config)
    return SyntheticInteractionScenario(config)


def comparison_specs() -> list[dict[str, Any]]:
    """Return deterministic uncertainty-comparison planner/predictor specs."""

    return [
        {
            "planner_name": "standard",
            "planner": StandardPlanner(),
            "predictor_name": "constant_velocity",
            "predictor": ConstantVelocityPredictor(),
        },
        {
            "planner_name": "uncertainty_aware_conservative",
            "planner": ConservativeUncertaintyPlanner(),
            "predictor_name": "synthetic_multimodal",
            "predictor": SyntheticMultimodalPredictor(),
        },
    ]


def _predict(
    predictor: TrajectoryPredictor,
    target_history: list[AgentState],
    horizon_steps: int,
    dt: float,
) -> PredictionSet:
    return predictor.predict(target_history, horizon_steps=horizon_steps, dt=dt)


def run_uncertainty_scenario(
    config: SyntheticScenarioConfig,
    planner_name: str,
    planner: Planner,
    predictor_name: str,
    predictor: TrajectoryPredictor,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one planner/predictor pair on one ambiguity scenario."""

    root = Path(output_root) if output_root is not None else project_root() / "results"
    metrics_dir = root / "metrics"
    logs_dir = root / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    scenario = build_uncertainty_scenario(config)
    state = scenario.reset()
    target_history = [state.agents[0]]

    log_rows: list[dict[str, Any]] = []
    distances: list[float] = []
    near_miss_seen = False
    collision_seen = False
    intervention_count = 0
    prediction_modes = 0

    for step in range(config.horizon_steps):
        target = state.agents[0]
        prediction = _predict(
            predictor,
            target_history,
            horizon_steps=config.horizon_steps,
            dt=config.dt,
        )
        prediction_modes = len(prediction.trajectories)
        action = planner.plan(state, [prediction])
        if action.acceleration < 0.0:
            intervention_count += 1

        distance = _instant_distance(state.ego, target)
        current_near_miss = is_near_miss(distance, threshold=NEAR_MISS_THRESHOLD)
        current_collision = is_collision(distance, threshold=COLLISION_THRESHOLD)
        time_to_collision = compute_time_to_collision(
            state.ego,
            target,
            collision_distance=COLLISION_THRESHOLD,
            max_time=config.horizon_steps * config.dt,
        )
        distances.append(distance)
        near_miss_seen = near_miss_seen or current_near_miss
        collision_seen = collision_seen or current_collision

        log_rows.append(
            {
                "step": step,
                "time": round(state.time_seconds, 6),
                "ego_x": round(state.ego.x, 6),
                "ego_y": round(state.ego.y, 6),
                "ego_vx": round(state.ego.vx, 6),
                "ego_vy": round(state.ego.vy, 6),
                "target_x": round(target.x, 6),
                "target_y": round(target.y, 6),
                "target_vx": round(target.vx, 6),
                "target_vy": round(target.vy, 6),
                "action_acceleration": round(action.acceleration, 6),
                "action_steering": round(action.steering, 6),
                "min_distance": round(distance, 6),
                "time_to_collision": "" if np.isinf(time_to_collision) else round(time_to_collision, 6),
                "near_miss": current_near_miss,
                "collision": current_collision,
            }
        )

        state = scenario.step(action)
        target_history.append(state.agents[0])

    log_file = logs_dir / f"uncertainty_comparison_{planner_name}_{config.name}.csv"
    metrics_file = metrics_dir / f"uncertainty_comparison_{planner_name}_{config.name}.json"
    _write_csv(log_file, UNCERTAINTY_LOG_COLUMNS, log_rows)

    summary = {
        "scenario": config.name,
        "planner": planner_name,
        "predictor": predictor_name,
        "prediction_modes": prediction_modes,
        "steps": config.horizon_steps,
        "final_time": round(config.horizon_steps * config.dt, 6),
        "min_distance": round(min(distances), 6),
        "near_miss": near_miss_seen,
        "collision": collision_seen,
        "intervention_count": intervention_count,
        "success": not collision_seen,
    }
    payload = {
        **summary,
        "metrics_file": str(metrics_file),
        "log_file": str(log_file),
    }
    metrics_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_uncertainty_planner_comparison(
    output_root: str | Path | None = None,
    configs: list[SyntheticScenarioConfig] | None = None,
) -> list[dict[str, Any]]:
    """Run deterministic versus uncertainty-aware planners on ambiguity scenarios."""

    root = Path(output_root) if output_root is not None else project_root() / "results"
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scenario_configs = configs if configs is not None else uncertainty_scenario_configs()

    results = [
        run_uncertainty_scenario(
            config,
            spec["planner_name"],
            spec["planner"],
            spec["predictor_name"],
            spec["predictor"],
            output_root=root,
        )
        for config in scenario_configs
        for spec in comparison_specs()
    ]

    summary_file = metrics_dir / "uncertainty_planner_comparison_summary.csv"
    _write_csv(
        summary_file,
        UNCERTAINTY_SUMMARY_COLUMNS,
        [{column: result[column] for column in UNCERTAINTY_SUMMARY_COLUMNS} for result in results],
    )
    return results


def main() -> None:
    """CLI entry point for synthetic uncertainty-aware planner comparison."""

    results = run_uncertainty_planner_comparison()
    print("Uncertainty planner comparison complete")
    print(
        json.dumps(
            [{column: result[column] for column in UNCERTAINTY_SUMMARY_COLUMNS} for result in results],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
