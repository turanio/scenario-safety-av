"""Run closed-loop baseline planning over synthetic scenarios."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from av_safety_eval.common.types import AgentState
from av_safety_eval.experiments.baseline_common import project_root
from av_safety_eval.metrics.safety import (
    compute_time_to_collision,
    is_collision,
    is_near_miss,
)
from av_safety_eval.planners.standard_planner import StandardPlanner
from av_safety_eval.predictors.constant_velocity import ConstantVelocityPredictor
from av_safety_eval.scenarios.synthetic_interaction import (
    SyntheticInteractionScenario,
    SyntheticScenarioConfig,
    baseline_matrix_configs,
)

CLOSED_LOOP_SUMMARY_COLUMNS = [
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

CLOSED_LOOP_LOG_COLUMNS = [
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


def _instant_distance(ego: AgentState, target: AgentState) -> float:
    return float(np.linalg.norm(ego.position - target.position))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_closed_loop_scenario(
    config: SyntheticScenarioConfig,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one closed-loop scenario and write per-step logs plus metrics."""

    root = Path(output_root) if output_root is not None else project_root() / "results"
    metrics_dir = root / "metrics"
    logs_dir = root / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    scenario = SyntheticInteractionScenario(config)
    predictor = ConstantVelocityPredictor()
    planner = StandardPlanner()
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
        current_near_miss = is_near_miss(distance, threshold=planner.near_miss_threshold)
        current_collision = is_collision(distance, threshold=planner.collision_threshold)
        time_to_collision = compute_time_to_collision(
            state.ego,
            target,
            collision_distance=planner.collision_threshold,
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

    min_distance = min(distances)
    log_file = logs_dir / f"closed_loop_{config.name}.csv"
    metrics_file = metrics_dir / f"closed_loop_{config.name}.json"
    _write_csv(log_file, CLOSED_LOOP_LOG_COLUMNS, log_rows)

    summary = {
        "scenario": config.name,
        "planner": "standard",
        "predictor": "constant_velocity",
        "steps": config.horizon_steps,
        "final_time": round(config.horizon_steps * config.dt, 6),
        "min_distance": round(min_distance, 6),
        "near_miss": near_miss_seen,
        "collision": collision_seen,
        "intervention_count": intervention_count,
        "success": not collision_seen,
    }
    metrics_payload = {
        **summary,
        "metrics_file": str(metrics_file),
        "log_file": str(log_file),
    }
    metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


def run_closed_loop_baseline(
    output_root: str | Path | None = None,
    configs: list[SyntheticScenarioConfig] | None = None,
) -> list[dict[str, Any]]:
    """Run all baseline matrix scenarios in closed loop."""

    root = Path(output_root) if output_root is not None else project_root() / "results"
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scenario_configs = configs if configs is not None else baseline_matrix_configs()
    results = [run_closed_loop_scenario(config, output_root=root) for config in scenario_configs]

    summary_file = metrics_dir / "closed_loop_baseline_summary.csv"
    _write_csv(
        summary_file,
        CLOSED_LOOP_SUMMARY_COLUMNS,
        [{column: result[column] for column in CLOSED_LOOP_SUMMARY_COLUMNS} for result in results],
    )
    return results


def main() -> None:
    """CLI entry point for closed-loop baseline evaluation."""

    results = run_closed_loop_baseline()
    print("Closed-loop baseline complete")
    print(
        json.dumps(
            [{column: result[column] for column in CLOSED_LOOP_SUMMARY_COLUMNS} for result in results],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
