"""Run controlled closed-loop CARLA validation with synthetic multimodal futures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from av_safety_eval.carla.carla_client import CarlaClientConfig, CarlaSession
from av_safety_eval.carla.image_capture import (
    CameraCaptureConfig,
    LatestRgbFrame,
    key_frame_filename,
)
from av_safety_eval.carla.metrics import CarlaPolicyMetrics, PolicyMetricTracker
from av_safety_eval.carla.scenarios import (
    HiddenRiskCarlaScenario,
    HiddenRiskScenarioConfig,
    HiddenRiskScenarioVariant,
    build_hidden_risk_scenario_suite,
)
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
SUITE_RESULT_FIELDS = (
    "scenario_name",
    "policy_name",
    "minimum_distance",
    "near_miss",
    "collision",
    "number_of_braking_interventions",
    "first_braking_time",
    "final_ego_speed",
    "scenario_success",
    "mode_probabilities",
    "actual_target_behavior",
    "interpretation",
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
        "--tick-timeout",
        type=float,
        default=60.0,
        help="Maximum seconds to wait for each synchronous CARLA tick.",
    )
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument(
        "--town",
        default=None,
        help="Optional CARLA town to load; Town04 or Town05 is recommended.",
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Hard upper bound on simulation steps for each policy.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every bounded CARLA tick as it starts.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the three controlled scenario variants and write suite outputs.",
    )
    parser.add_argument(
        "--image-timeout",
        type=float,
        default=5.0,
        help="Maximum seconds to wait for each selected RGB camera frame.",
    )
    parser.add_argument(
        "--no-image-capture",
        action="store_true",
        help="Disable elevated RGB key-frame capture.",
    )
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


def _capture_key_frame(
    receiver: LatestRgbFrame,
    carla_frame: int,
    output_path: Path,
    timeout_seconds: float,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[CARLA] warning: image exists, skipping {output_path}")
        return
    try:
        receiver.save_frame(carla_frame, output_path, timeout_seconds)
        print(f"[CARLA] saved image {output_path}")
    except Exception as exc:
        print(f"[CARLA] warning: image capture failed for {output_path}: {exc}")


def _run_policy(
    session: CarlaSession,
    config: HiddenRiskScenarioConfig,
    policy_name: str,
    evaluator: Callable[[np.ndarray, np.ndarray], SafetyFilterResult],
    max_steps: int,
    scenario_name: str,
    image_output_dir: Path | None = None,
    capture_config: CameraCaptureConfig | None = None,
    overwrite_images: bool = False,
) -> tuple[CarlaPolicyMetrics, list[float], list[float]]:
    scenario = HiddenRiskCarlaScenario(config)
    controller = SimpleLongitudinalController(target_speed_mps=config.ego_target_speed_mps)
    tracker = PolicyMetricTracker(config.near_miss_threshold_m)
    actors = None
    frame_receiver = None

    print(f"[CARLA] starting policy {policy_name}")
    try:
        actors = scenario.spawn(session)
        print(f"[CARLA] spawn completed for {policy_name}")
        if image_output_dir is not None and capture_config is not None:
            try:
                frame_receiver = LatestRgbFrame()
                session.attach_rgb_camera(
                    actors.ego,
                    frame_receiver,
                    capture_config,
                )
                print(f"[CARLA] elevated RGB camera attached for {policy_name}")
            except Exception as exc:
                frame_receiver = None
                print(
                    f"[CARLA] warning: RGB camera unavailable for "
                    f"{scenario_name}/{policy_name}: {exc}"
                )
        for step in range(config.num_steps):
            if step >= max_steps:
                raise RuntimeError(
                    f"Policy {policy_name} reached the maximum of {max_steps} "
                    "simulation steps before completing."
                )
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
            carla_frame = session.tick()
            if (
                frame_receiver is not None
                and capture_config is not None
                and image_output_dir is not None
                and step in capture_config.capture_steps
            ):
                image_path = image_output_dir / key_frame_filename(
                    scenario_name,
                    policy_name,
                    step,
                )
                _capture_key_frame(
                    frame_receiver,
                    carla_frame,
                    image_path,
                    capture_config.image_timeout_seconds,
                    overwrite_images,
                )
            tracker.update(
                time_seconds=next_time,
                center_distance_m=scenario.center_distance(actors.ego, actors.target),
                ego_speed_mps=_speed_mps(actors.ego),
                braking=should_brake,
                collision_detected=actors.collision_recorder.collision_detected,
            )
            completed_steps = step + 1
            if completed_steps % 20 == 0 or completed_steps == config.num_steps:
                print(
                    f"[CARLA] {policy_name}: completed step "
                    f"{completed_steps}/{config.num_steps}"
                )

        metrics = tracker.finalize(
            policy_name=policy_name,
            expected_steps=config.num_steps,
            collision_sensor_available=actors.collision_recorder.available,
            collision_note=actors.collision_recorder.note,
        )
        print(f"[CARLA] finished policy {policy_name}")
        return metrics, tracker.times, tracker.distances
    finally:
        session.destroy_actors()


def run_experiment(
    client_config: CarlaClientConfig,
    scenario_config: HiddenRiskScenarioConfig,
    max_steps: int = 2000,
    image_output_dir: Path | None = None,
    capture_config: CameraCaptureConfig | None = None,
    overwrite_images: bool = False,
) -> tuple[list[CarlaPolicyMetrics], dict[str, tuple[list[float], list[float]]]]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if scenario_config.num_steps > max_steps:
        raise ValueError(
            f"Scenario requires {scenario_config.num_steps} steps, exceeding the "
            f"configured maximum of {max_steps}."
        )

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
                max_steps,
                "hidden_risk",
                image_output_dir,
                capture_config,
                overwrite_images,
            )
            metrics.append(result)
            distance_series[policy_name] = (times, distances)
    return metrics, distance_series


def run_scenario_suite(
    client_config: CarlaClientConfig,
    variants: tuple[HiddenRiskScenarioVariant, ...],
    max_steps: int = 2000,
    image_output_dir: Path | None = None,
    capture_config: CameraCaptureConfig | None = None,
    overwrite_images: bool = False,
) -> tuple[
    list[tuple[HiddenRiskScenarioVariant, CarlaPolicyMetrics]],
    dict[str, dict[str, tuple[list[float], list[float]]]],
]:
    if not variants:
        raise ValueError("variants must not be empty")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    for variant in variants:
        if variant.config.num_steps > max_steps:
            raise ValueError(
                f"Scenario {variant.name} requires {variant.config.num_steps} "
                f"steps, exceeding the configured maximum of {max_steps}."
            )

    results: list[tuple[HiddenRiskScenarioVariant, CarlaPolicyMetrics]] = []
    suite_series: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    with CarlaSession(client_config) as session:
        for variant in variants:
            print(f"[CARLA] starting scenario {variant.name}")
            evaluators = _policy_evaluators(variant.config)
            scenario_series: dict[str, tuple[list[float], list[float]]] = {}
            for policy_name in POLICY_NAMES:
                metric, times, distances = _run_policy(
                    session,
                    variant.config,
                    policy_name,
                    evaluators[policy_name],
                    max_steps,
                    variant.name,
                    image_output_dir,
                    capture_config,
                    overwrite_images,
                )
                results.append((variant, metric))
                scenario_series[policy_name] = (times, distances)
            suite_series[variant.name] = scenario_series
            print(f"[CARLA] finished scenario {variant.name}")
    return results, suite_series


def _output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "hidden_risk_results.csv",
        output_dir / "hidden_risk_summary.md",
        output_dir / "distance_over_time.png",
    )


def _suite_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "carla_scenario_suite_results.csv",
        output_dir / "carla_scenario_suite_summary.md",
        output_dir / "carla_scenario_suite_distance_over_time.png",
    )


def _prepare_outputs(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    return _prepare_paths(_output_paths(output_dir), overwrite)


def _prepare_paths(
    paths: tuple[Path, Path, Path],
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "CARLA outputs already exist. Use --overwrite to replace them:\n" + listing
        )
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    return paths


def _write_results(path: Path, metrics: list[CarlaPolicyMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric.to_dict() for metric in metrics)


def _write_suite_results(
    path: Path,
    results: list[tuple[HiddenRiskScenarioVariant, CarlaPolicyMetrics]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUITE_RESULT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for variant, metric in results:
            metric_row = metric.to_dict()
            writer.writerow(
                {
                    "scenario_name": variant.name,
                    "policy_name": metric.policy_name,
                    "minimum_distance": metric.minimum_distance,
                    "near_miss": metric.near_miss,
                    "collision": metric.collision,
                    "number_of_braking_interventions": (
                        metric.number_of_braking_interventions
                    ),
                    "first_braking_time": metric_row["first_braking_time"],
                    "final_ego_speed": metric.final_ego_speed,
                    "scenario_success": metric.scenario_success,
                    "mode_probabilities": json.dumps(
                        list(variant.config.mode_probabilities)
                    ),
                    "actual_target_behavior": (
                        variant.config.actual_target_behavior
                    ),
                    "interpretation": variant.interpretation,
                }
            )


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


def _plot_suite_distances(
    path: Path,
    variants: tuple[HiddenRiskScenarioVariant, ...],
    suite_series: dict[str, dict[str, tuple[list[float], list[float]]]],
) -> None:
    colors = {
        "top1_policy": "#1f4e79",
        "worst_case_policy": "#b22222",
        "probability_aware_policy": "#2e7d32",
    }
    fig, axes = plt.subplots(
        len(variants),
        1,
        figsize=(9.5, 4.0 * len(variants)),
        sharex=True,
        squeeze=False,
    )
    for axis, variant in zip(axes[:, 0], variants):
        for policy_name in POLICY_NAMES:
            times, distances = suite_series[variant.name][policy_name]
            axis.plot(
                times,
                distances,
                linewidth=2.0,
                color=colors[policy_name],
                label=policy_name,
            )
        axis.axhline(
            variant.config.near_miss_threshold_m,
            color="#d97706",
            linestyle=":",
            linewidth=1.5,
            label=(
                "Near-miss threshold "
                f"({variant.config.near_miss_threshold_m:.1f} m)"
            ),
        )
        axis.set_title(variant.name.replace("_", " ").title())
        axis.set_ylabel("Center distance (m)")
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=True, fontsize=8)
    axes[-1, 0].set_xlabel("Simulation time (s)")
    fig.suptitle("Controlled CARLA scenario-suite policy comparison", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Scripted synthetic futures; controlled closed-loop behavior without online QCNet.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _summary_markdown(
    metrics: list[CarlaPolicyMetrics],
    client_config: CarlaClientConfig,
    scenario_config: HiddenRiskScenarioConfig,
    image_capture_enabled: bool = True,
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
            "## CARLA Visual Evidence",
            "",
            (
                "Successfully captured elevated RGB key frames are saved in the "
                "output directory's `images/` subdirectory at rollout steps 0, 40, "
                "70, and 100. Filenames identify the scenario, policy, and step."
                if image_capture_enabled
                else "RGB key-frame capture was disabled for this run."
            ),
            "",
            (
                "A camera timeout or save failure is reported as a warning and does "
                "not alter the numerical policy metrics."
                if image_capture_enabled
                else ""
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


def _suite_summary_markdown(
    results: list[tuple[HiddenRiskScenarioVariant, CarlaPolicyMetrics]],
    variants: tuple[HiddenRiskScenarioVariant, ...],
    client_config: CarlaClientConfig,
    image_capture_enabled: bool = True,
) -> str:
    rows = [
        "# Controlled CARLA Scenario Suite",
        "",
        "## Scope",
        "",
        (
            "This small suite is controlled closed-loop CARLA validation inspired "
            "by interaction patterns identified in the QCNet/Argoverse 2 analysis. "
            "It does not run QCNet online and is not intended as a general CARLA "
            "benchmark."
        ),
        "",
        (
            "The QCNet/AV2 500-scenario evaluation provides the larger-scale "
            "open-loop evidence. These CARLA variants test whether differences "
            "between top-1, worst-case, and probability-aware policies affect "
            "closed-loop vehicle behavior under controlled conditions."
        ),
        "",
        "## Configuration",
        "",
        f"- CARLA endpoint: `{client_config.host}:{client_config.port}`",
        "- Synchronous mode: `true`",
        f"- Fixed delta: `{client_config.fixed_delta_seconds:.2f} s`",
        "- Probability-aware cutoff: `p >= 0.05`",
        "- Distance metric: ego-target center distance",
        "- Future hypotheses: scripted; no online QCNet inference",
        "",
        "## Results by Scenario",
        "",
    ]
    for variant in variants:
        probabilities = ", ".join(
            f"{value:.2f}" for value in variant.config.mode_probabilities
        )
        rows.extend(
            [
                f"### `{variant.name}`",
                "",
                f"Inspired by: {variant.inspired_by}.",
                "",
                f"Mode probabilities (safe, moderate, aggressive): `{probabilities}`.",
                "",
                f"Actual target behavior: `{variant.config.actual_target_behavior}`.",
                "",
                variant.interpretation,
                "",
                (
                    "| Policy | Minimum distance (m) | Near miss | Collision | "
                    "Brake interventions | First brake (s) | Final speed (m/s) | Success |"
                ),
                "|---|---:|---|---|---:|---:|---:|---|",
            ]
        )
        for result_variant, metric in results:
            if result_variant.name != variant.name:
                continue
            first_brake = (
                "-"
                if metric.first_braking_time is None
                else f"{metric.first_braking_time:.2f}"
            )
            rows.append(
                f"| `{metric.policy_name}` | {metric.minimum_distance:.3f} | "
                f"{str(metric.near_miss).lower()} | "
                f"{str(metric.collision).lower()} | "
                f"{metric.number_of_braking_interventions} | {first_brake} | "
                f"{metric.final_ego_speed:.3f} | "
                f"{str(metric.scenario_success).lower()} |"
            )
        rows.append("")

    rows.extend(
        [
            "## CARLA Visual Evidence",
            "",
            (
                "Successfully captured elevated RGB key frames are saved in the "
                "output directory's `images/` subdirectory at rollout steps 0, 40, "
                "70, and 100. Filenames identify the scenario, policy, and step."
                if image_capture_enabled
                else "RGB key-frame capture was disabled for this run."
            ),
            "",
            (
                "The views are attached above the ego vehicle to keep the ego-target "
                "interaction visible. Capture failures produce warnings without "
                "stopping the controlled rollout."
                if image_capture_enabled
                else ""
            ),
            "",
            "## Limitations",
            "",
            (
                "The three variants are deliberately small and scripted. Minimum "
                "distance is center-to-center distance rather than oriented vehicle-"
                "footprint clearance. Collision is reported only when CARLA's "
                "collision sensor records an event."
            ),
            "",
            (
                "The results can show policy-dependent closed-loop behavior in these "
                "controlled cases, but they do not prove collision avoidance, general "
                "safety improvement, or performance of online QCNet in CARLA."
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
        tick_timeout_seconds=args.tick_timeout,
        town=args.town,
        fixed_delta_seconds=scenario_config.fixed_delta_seconds,
        traffic_manager_port=args.traffic_manager_port,
        verbose=args.verbose,
    )
    capture_config = (
        None
        if args.no_image_capture
        else CameraCaptureConfig(image_timeout_seconds=args.image_timeout)
    )
    image_output_dir = (
        None if capture_config is None else args.output_dir / "images"
    )
    if args.suite:
        paths = _suite_output_paths(args.output_dir)
        results_csv, summary_md, distance_png = _prepare_paths(
            paths, args.overwrite
        )
        variants = build_hidden_risk_scenario_suite(args.duration)
        results, series = run_scenario_suite(
            client_config,
            variants,
            max_steps=args.max_steps,
            image_output_dir=image_output_dir,
            capture_config=capture_config,
            overwrite_images=args.overwrite,
        )
        _write_suite_results(results_csv, results)
        _plot_suite_distances(distance_png, variants, series)
        summary_md.write_text(
            _suite_summary_markdown(
                results,
                variants,
                client_config,
                image_capture_enabled=capture_config is not None,
            ),
            encoding="utf-8",
        )
    else:
        results_csv, summary_md, distance_png = _prepare_outputs(
            args.output_dir, args.overwrite
        )
        metrics, series = run_experiment(
            client_config,
            scenario_config,
            max_steps=args.max_steps,
            image_output_dir=image_output_dir,
            capture_config=capture_config,
            overwrite_images=args.overwrite,
        )
        _write_results(results_csv, metrics)
        _plot_distances(distance_png, series, scenario_config.near_miss_threshold_m)
        summary_md.write_text(
            _summary_markdown(
                metrics,
                client_config,
                scenario_config,
                image_capture_enabled=capture_config is not None,
            ),
            encoding="utf-8",
        )

    print(f"Created {results_csv}")
    print(f"Created {summary_md}")
    print(f"Created {distance_png}")
    if image_output_dir is not None:
        print(f"CARLA key frames saved under {image_output_dir} when capture succeeded")


if __name__ == "__main__":
    main()
