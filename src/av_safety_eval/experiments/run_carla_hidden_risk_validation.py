"""Run controlled closed-loop CARLA validation with synthetic multimodal futures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from av_safety_eval.carla.carla_client import CarlaClientConfig, CarlaSession
from av_safety_eval.carla.metrics import CarlaPolicyMetrics, PolicyMetricTracker
from av_safety_eval.carla.scenarios import HiddenRiskCarlaScenario, HiddenRiskScenarioConfig
from av_safety_eval.carla.vehicle_controller import SimpleLongitudinalController
from av_safety_eval.planning.safety_filter import (
    BRAKE,
    SafetyFilterResult,
    evaluate_probability_aware_filter,
    evaluate_top1_filter,
    evaluate_worst_case_filter,
)


POLICY_NAMES = ("top1_policy", "worst_case_policy", "probability_aware_policy")
RESULT_FIELDS = (
    "policy_name",
    "minimum_distance",
    "near_miss",
    "collision",
    "collision_note",
    "number_of_braking_interventions",
    "first_braking_time",
    "final_ego_speed",
    "scenario_success",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled CARLA cut-in with top-1, worst-case, and "
            "probability-aware synthetic-future policies."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--town",
        default=None,
        help="Optional CARLA town to load; Town04 or Town05 is recommended.",
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/carla_validation"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing CARLA validation outputs.",
    )
    return parser.parse_args()


def _policy_evaluators(
    config: HiddenRiskScenarioConfig,
) -> dict[str, Callable[[np.ndarray, np.ndarray], SafetyFilterResult]]:
    return {
        "top1_policy": lambda distances, probabilities: evaluate_top1_filter(
            distances,
            probabilities,
            safety_threshold_m=config.near_miss_threshold_m,
        ),
        "worst_case_policy": lambda distances, probabilities: evaluate_worst_case_filter(
            distances,
            probabilities,
            safety_threshold_m=config.near_miss_threshold_m,
        ),
        "probability_aware_policy": (
            lambda distances, probabilities: evaluate_probability_aware_filter(
                distances,
                probabilities,
                safety_threshold_m=config.near_miss_threshold_m,
                probability_threshold=config.probability_threshold,
            )
        ),
    }


def _speed_mps(vehicle: object) -> float:
    velocity = vehicle.get_velocity()
    return float(math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))


def _run_policy(
    session: CarlaSession,
    config: HiddenRiskScenarioConfig,
    policy_name: str,
    evaluator: Callable[[np.ndarray, np.ndarray], SafetyFilterResult],
) -> tuple[CarlaPolicyMetrics, list[float], list[float]]:
    scenario = HiddenRiskCarlaScenario(config)
    actors = scenario.spawn(session)
    controller = SimpleLongitudinalController(target_speed_mps=config.ego_target_speed_mps)
    tracker = PolicyMetricTracker(config.near_miss_threshold_m)

    try:
        for step in range(config.num_steps):
            time_seconds = step * config.fixed_delta_seconds
            modes = scenario.synthetic_future_modes(time_seconds)
            ego_future = scenario.ego_constant_velocity_future(actors.ego)
            per_mode_distances = np.linalg.norm(
                modes.positions - ego_future[None, :, :], axis=-1
            )
            decision = evaluator(per_mode_distances, modes.probabilities)
            should_brake = decision.action == BRAKE
            command = controller.command(_speed_mps(actors.ego), should_brake=should_brake)
            actors.ego.apply_control(
                session.carla.VehicleControl(
                    throttle=command.throttle,
                    brake=command.brake,
                    steer=command.steer,
                )
            )

            next_time = (step + 1) * config.fixed_delta_seconds
            scenario.update_target(actors.target, next_time, session)
            session.tick()
            tracker.update(
                time_seconds=next_time,
                center_distance_m=scenario.center_distance(actors.ego, actors.target),
                ego_speed_mps=_speed_mps(actors.ego),
                braking=should_brake,
                collision_detected=actors.collision_recorder.collision_detected,
            )

        metrics = tracker.finalize(
            policy_name=policy_name,
            expected_steps=config.num_steps,
            collision_sensor_available=actors.collision_recorder.available,
            collision_note=actors.collision_recorder.note,
        )
        return metrics, tracker.times, tracker.distances
    finally:
        session.destroy_actors()
        session.tick()


def run_experiment(
    client_config: CarlaClientConfig,
    scenario_config: HiddenRiskScenarioConfig,
) -> tuple[list[CarlaPolicyMetrics], dict[str, tuple[list[float], list[float]]]]:
    metrics: list[CarlaPolicyMetrics] = []
    distance_series: dict[str, tuple[list[float], list[float]]] = {}
    evaluators = _policy_evaluators(scenario_config)

    with CarlaSession(client_config) as session:
        for policy_name in POLICY_NAMES:
            result, times, distances = _run_policy(
                session,
                scenario_config,
                policy_name,
                evaluators[policy_name],
            )
            metrics.append(result)
            distance_series[policy_name] = (times, distances)
    return metrics, distance_series


def _output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "hidden_risk_results.csv",
        output_dir / "hidden_risk_summary.md",
        output_dir / "distance_over_time.png",
    )


def _prepare_outputs(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    paths = _output_paths(output_dir)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "CARLA outputs already exist. Use --overwrite to replace them:\n" + listing
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _write_results(path: Path, metrics: list[CarlaPolicyMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric.to_dict() for metric in metrics)


def _plot_distances(
    path: Path,
    series: dict[str, tuple[list[float], list[float]]],
    threshold_m: float,
) -> None:
    colors = {
        "top1_policy": "#1f4e79",
        "worst_case_policy": "#b22222",
        "probability_aware_policy": "#2e7d32",
    }
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for policy_name in POLICY_NAMES:
        times, distances = series[policy_name]
        ax.plot(
            times,
            distances,
            linewidth=2.2,
            color=colors[policy_name],
            label=policy_name,
        )
    ax.axhline(
        threshold_m,
        color="#d97706",
        linestyle=":",
        linewidth=1.6,
        label=f"Near-miss threshold ({threshold_m:.1f} m)",
    )
    ax.set_title("Controlled CARLA hidden-risk interaction")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Ego-target center distance (m)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.text(
        0.5,
        0.015,
        "Closed-loop CARLA control with scripted synthetic futures; no online QCNet inference.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _summary_markdown(
    metrics: list[CarlaPolicyMetrics],
    client_config: CarlaClientConfig,
    scenario_config: HiddenRiskScenarioConfig,
) -> str:
    rows = [
        "# Controlled CARLA Hidden-Risk Validation",
        "",
        "## Scope",
        "",
        (
            "This experiment is a controlled closed-loop CARLA validation of a "
            "cut-in pattern inspired by the QCNet/AV2 hidden-risk analysis. QCNet is "
            "not executed online in CARLA; all three future hypotheses are scripted "
            "and deterministic."
        ),
        "",
        "## Configuration",
        "",
        f"- CARLA endpoint: `{client_config.host}:{client_config.port}`",
        "- Synchronous mode: `true`",
        f"- Fixed delta: `{client_config.fixed_delta_seconds:.2f} s`",
        f"- Scenario duration: `{scenario_config.duration_seconds:.1f} s`",
        f"- Center-distance threshold: `{scenario_config.near_miss_threshold_m:.1f} m`",
        f"- Probability-aware cutoff: `p >= {scenario_config.probability_threshold:.2f}`",
        "- Actual target behavior: scripted moderate cut-in",
        "- Hypotheses: safe continuation, moderate cut-in, aggressive cut-in",
        "- Mode probabilities: `0.78, 0.18, 0.04`",
        "- The probability-aware policy includes the safe and moderate modes",
        "- Braking interventions count transitions into a braking episode",
        (
            "- Scenario success means the rollout completed all expected steps; "
            "an available collision-sensor event makes it false"
        ),
        "",
        "## Results",
        "",
        (
            "| Policy | Minimum distance (m) | Near miss | Collision sensor | "
            "Brake interventions | First brake (s) | Final ego speed (m/s) | Success |"
        ),
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for metric in metrics:
        first_brake = (
            "-" if metric.first_braking_time is None else f"{metric.first_braking_time:.2f}"
        )
        rows.append(
            f"| `{metric.policy_name}` | {metric.minimum_distance:.3f} | "
            f"{str(metric.near_miss).lower()} | {str(metric.collision).lower()} | "
            f"{metric.number_of_braking_interventions} | {first_brake} | "
            f"{metric.final_ego_speed:.3f} | {str(metric.scenario_success).lower()} |"
        )
    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The three rows compare how mode selection changes the timing and "
                "frequency of braking under identical initial conditions. Differences "
                "in minimum distance are outcomes of this scripted CARLA interaction; "
                "they do not establish a general safety improvement."
            ),
            "",
            "## Limitations",
            "",
            (
                "The forecast modes are synthetic and scenario-specific. The reported "
                "minimum distance is center-to-center distance, not oriented "
                "vehicle-footprint clearance. Collision is true only when the CARLA "
                "collision sensor records an event; if that sensor is unavailable, the "
                "CSV reports false with an explicit note."
            ),
            "",
            (
                "This is one controlled closed-loop experiment, not evidence of "
                "collision avoidance, a full autonomous-driving stack, online QCNet "
                "integration, or general closed-loop safety improvement."
            ),
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    scenario_config = HiddenRiskScenarioConfig(duration_seconds=args.duration)
    client_config = CarlaClientConfig(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout,
        town=args.town,
        fixed_delta_seconds=scenario_config.fixed_delta_seconds,
    )
    results_csv, summary_md, distance_png = _prepare_outputs(
        args.output_dir, args.overwrite
    )
    metrics, series = run_experiment(client_config, scenario_config)
    _write_results(results_csv, metrics)
    _plot_distances(distance_png, series, scenario_config.near_miss_threshold_m)
    summary_md.write_text(
        _summary_markdown(metrics, client_config, scenario_config),
        encoding="utf-8",
    )

    print(f"Created {results_csv}")
    print(f"Created {summary_md}")
    print(f"Created {distance_png}")


if __name__ == "__main__":
    main()
