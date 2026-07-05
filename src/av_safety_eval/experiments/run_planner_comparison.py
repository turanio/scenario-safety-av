"""Compare deterministic planners over the synthetic closed-loop scenarios."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from av_safety_eval.common.types import AgentState
from av_safety_eval.experiments.baseline_common import project_root
from av_safety_eval.metrics.safety import compute_time_to_collision, is_collision, is_near_miss
from av_safety_eval.planners.base import Planner
from av_safety_eval.planners.naive_planner import NaivePlanner
from av_safety_eval.planners.standard_planner import StandardPlanner
from av_safety_eval.predictors.constant_velocity import ConstantVelocityPredictor
from av_safety_eval.scenarios.synthetic_interaction import (
    SyntheticInteractionScenario,
    SyntheticScenarioConfig,
    baseline_matrix_configs,
)

PLANNER_COMPARISON_SUMMARY_COLUMNS = [
    "scenario",
    "planner",
    "predictor",
    "steps",
    "final_time",
    "min_distance",
    "near_miss",
    "collision",
    "intervention_count",
    "success",
]

PLANNER_COMPARISON_LOG_COLUMNS = [
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


def planner_specs() -> dict[str, Planner]:
    """Return planners included in the deterministic comparison."""

    return {
        "naive": NaivePlanner(),
        "standard": StandardPlanner(),
    }


def _instant_distance(ego: AgentState, target: AgentState) -> float:
    return float(np.linalg.norm(ego.position - target.position))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_planner_scenario(
    config: SyntheticScenarioConfig,
    planner_name: str,
    planner: Planner,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one planner on one synthetic scenario and persist artifacts."""

    root = Path(output_root) if output_root is not None else project_root() / "results"
    metrics_dir = root / "metrics"
    logs_dir = root / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    scenario = SyntheticInteractionScenario(config)
    predictor = ConstantVelocityPredictor()
    state = scenario.reset()

    log_rows: list[dict[str, Any]] = []
    distances: list[float] = []
    near_miss_seen = False
    collision_seen = False
    intervention_count = 0

    for step in range(config.horizon_steps):
        target = state.agents[0]
        prediction = predictor.predict(
            [target],
            horizon_steps=config.horizon_steps,
            dt=config.dt,
        )
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

    log_file = logs_dir / f"planner_comparison_{planner_name}_{config.name}.csv"
    metrics_file = metrics_dir / f"planner_comparison_{planner_name}_{config.name}.json"
    _write_csv(log_file, PLANNER_COMPARISON_LOG_COLUMNS, log_rows)

    summary = {
        "scenario": config.name,
        "planner": planner_name,
        "predictor": "constant_velocity",
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


def run_planner_comparison(
    output_root: str | Path | None = None,
    configs: list[SyntheticScenarioConfig] | None = None,
) -> list[dict[str, Any]]:
    """Run naive and standard planners on all synthetic scenarios."""

    root = Path(output_root) if output_root is not None else project_root() / "results"
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scenario_configs = configs if configs is not None else baseline_matrix_configs()
    planners = planner_specs()

    results = [
        run_planner_scenario(config, planner_name, planner, output_root=root)
        for planner_name, planner in planners.items()
        for config in scenario_configs
    ]

    summary_file = metrics_dir / "planner_comparison_summary.csv"
    _write_csv(
        summary_file,
        PLANNER_COMPARISON_SUMMARY_COLUMNS,
        [{column: result[column] for column in PLANNER_COMPARISON_SUMMARY_COLUMNS} for result in results],
    )
    return results


def main() -> None:
    """CLI entry point for planner comparison."""

    results = run_planner_comparison()
    print("Planner comparison complete")
    print(
        json.dumps(
            [
                {column: result[column] for column in PLANNER_COMPARISON_SUMMARY_COLUMNS}
                for result in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
