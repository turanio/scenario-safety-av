import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


SELECTED_SCENARIOS = (
    {
        "scenario_id": "001749f1-bc1c-47fb-a13f-9ab1f2c050a8",
        "scenario_type": "Hidden risk",
        "slug": "hidden_risk_001749",
    },
    {
        "scenario_id": "0091bad9-e7b2-4c07-aa12-6b5fd03c63d2",
        "scenario_type": "High-confidence close interaction",
        "slug": "high_confidence_close_0091bad",
    },
    {
        "scenario_id": "00351569-255c-433e-b97b-e2a844d1b6e0",
        "scenario_type": "Real near-miss",
        "slug": "real_near_miss_003515",
    },
)

CSV_FIELDS = (
    "scenario_id",
    "scenario_type",
    "top1_min_distance",
    "worst_case_min_distance",
    "ground_truth_min_distance",
    "timestep_of_top1_min",
    "timestep_of_worst_case_min",
    "timestep_of_ground_truth_min",
    "top1_mode",
    "worst_case_mode",
    "top1_probability",
    "worst_case_mode_probability",
)

CANDIDATE_CSV_FIELDS = (
    "scenario_id",
    "candidate_type",
    "top1_min_distance",
    "worst_case_min_distance",
    "ground_truth_min_distance",
    "timestep_of_top1_min",
    "timestep_of_worst_case_min",
    "timestep_of_ground_truth_min",
    "top1_mode",
    "worst_case_mode",
    "top1_probability",
    "worst_case_mode_probability",
    "min_occurs_at_horizon_end",
    "recommended_use",
    "notes",
)

RECOMMENDED_USES = {
    "primary_case_study",
    "secondary_case",
    "appendix_only",
    "reject",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create open-loop map and actor-context validation plots for selected "
            "QCNet scenarios."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("results/qcnet_batch/artifacts"),
    )
    parser.add_argument(
        "--ranking-json",
        type=Path,
        default=Path("results/qcnet_batch/qcnet_batch_scenario_ranking.json"),
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=Path("../data/argoverse2/val/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qcnet_batch/scenario_validation"),
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=None,
        help="Optional JSON list of scenario definitions; defaults to the original three.",
    )
    parser.add_argument(
        "--csv-name",
        default=None,
        help="Output CSV filename; selected automatically when omitted.",
    )
    parser.add_argument("--zoom-radius", type=float, default=15.0)
    return parser.parse_args()


def load_scenario_definitions(path: Path | None) -> list[dict]:
    if path is None:
        return list(SELECTED_SCENARIOS)
    with path.open(encoding="utf-8") as handle:
        scenarios = json.load(handle)
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"Expected a non-empty scenario list in {path}")

    required = {"scenario_id", "candidate_type", "slug", "recommended_use", "notes"}
    for scenario in scenarios:
        missing = required - set(scenario)
        if missing:
            raise ValueError(
                f"Scenario definition {scenario.get('scenario_id', '<unknown>')} "
                f"is missing: {sorted(missing)}"
            )
        if scenario["recommended_use"] not in RECOMMENDED_USES:
            raise ValueError(
                f"Unsupported recommended_use: {scenario['recommended_use']}"
            )
    return scenarios


def scenario_label(metadata: dict) -> str:
    return str(metadata.get("scenario_type", metadata.get("candidate_type", "Scenario")))


def load_ranking(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in ranking JSON: {path}")
    return {row["scenario_id"]: row for row in rows}


def joint_future_mask(data: np.lib.npyio.NpzFile, horizon: int) -> np.ndarray:
    if "ego_future_valid_mask" in data and "target_future_valid_mask" in data:
        return (
            data["ego_future_valid_mask"][:horizon].astype(bool)
            & data["target_future_valid_mask"][:horizon].astype(bool)
        )
    return np.ones(horizon, dtype=bool)


def mask_trajectory(trajectory: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = trajectory.astype(float).copy()
    masked[~mask] = np.nan
    return masked


def distance_series(
    trajectory: np.ndarray, ego_future: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    distances = np.linalg.norm(trajectory - ego_future, axis=-1).astype(float)
    distances[~mask] = np.nan
    return distances


def analyze_artifact(path: Path, expected_scenario_id: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        scenario_id = str(data["scenario_id"])
        if scenario_id != expected_scenario_id:
            raise ValueError(
                f"Artifact scenario ID {scenario_id} does not match {expected_scenario_id}"
            )

        positions = data["positions"].astype(float)
        probabilities = data["probabilities"].astype(float)
        ego_future = data["ego_future_positions"].astype(float)
        target_future = data["target_future_positions"].astype(float)
        horizon = min(positions.shape[1], len(ego_future), len(target_future))
        mask = joint_future_mask(data, horizon)
        if not np.any(mask):
            raise ValueError(f"No jointly valid future steps in {path}")

        positions = positions[:, :horizon]
        ego_future = ego_future[:horizon]
        target_future = target_future[:horizon]
        per_mode_distances = np.stack(
            [distance_series(mode, ego_future, mask) for mode in positions]
        )
        per_mode_minima = np.nanmin(per_mode_distances, axis=1)
        top1_mode = int(np.argmax(probabilities))
        worst_case_mode = int(np.nanargmin(per_mode_minima))
        ground_truth_distances = distance_series(target_future, ego_future, mask)

        ego_history = data["ego_history_positions"].astype(float)
        target_history = data["target_history_positions"].astype(float)
        if "ego_history_valid_mask" in data:
            ego_history = mask_trajectory(
                ego_history, data["ego_history_valid_mask"].astype(bool)
            )
        if "target_history_valid_mask" in data:
            target_history = mask_trajectory(
                target_history, data["target_history_valid_mask"].astype(bool)
            )

        result = {
            "scenario_id": scenario_id,
            "dt": float(data["dt"]),
            "positions": positions,
            "probabilities": probabilities,
            "ego_history": ego_history,
            "ego_future": mask_trajectory(ego_future, mask),
            "target_history": target_history,
            "target_future": mask_trajectory(target_future, mask),
            "top1_mode": top1_mode,
            "worst_case_mode": worst_case_mode,
            "top1_distances": per_mode_distances[top1_mode],
            "worst_case_distances": per_mode_distances[worst_case_mode],
            "ground_truth_distances": ground_truth_distances,
        }

    result["timestep_of_top1_min"] = int(
        np.nanargmin(result["top1_distances"])
    )
    result["timestep_of_worst_case_min"] = int(
        np.nanargmin(result["worst_case_distances"])
    )
    result["timestep_of_ground_truth_min"] = int(
        np.nanargmin(result["ground_truth_distances"])
    )
    result["top1_min_distance"] = float(np.nanmin(result["top1_distances"]))
    result["worst_case_min_distance"] = float(
        np.nanmin(result["worst_case_distances"])
    )
    result["ground_truth_min_distance"] = float(
        np.nanmin(result["ground_truth_distances"])
    )
    return result


def validate_ranking(analysis: dict, ranking_row: dict) -> None:
    expected_values = {
        "top1_min_distance": float(ranking_row["top1_min_distance"]),
        "worst_case_min_distance": float(ranking_row["worst_case_min_distance"]),
        "ground_truth_min_distance": float(ranking_row["ground_truth_min_distance"]),
    }
    for field, expected in expected_values.items():
        actual = float(analysis[field])
        if not np.isclose(actual, expected, rtol=1e-5, atol=1e-5):
            raise ValueError(
                f"Artifact and ranking disagree for {field}: {actual} != {expected}"
            )
    if analysis["top1_mode"] != int(ranking_row["top1_mode"]):
        raise ValueError("Artifact and ranking disagree on the top-1 mode")


def map_path_for_scenario(map_root: Path, scenario_id: str) -> Path | None:
    scenario_dir = map_root / scenario_id
    expected = scenario_dir / f"log_map_archive_{scenario_id}.json"
    if expected.is_file():
        return expected
    matches = sorted(scenario_dir.glob("log_map_archive_*.json"))
    return matches[0] if matches else None


def load_map(path: Path | None) -> dict | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def point_array(points: list[dict]) -> np.ndarray:
    return np.asarray([[point["x"], point["y"]] for point in points], dtype=float)


def draw_map_context(ax: plt.Axes, map_data: dict | None) -> None:
    if map_data is None:
        ax.text(
            0.02,
            0.02,
            "AV2 map context unavailable",
            transform=ax.transAxes,
            fontsize=8,
            color="#6b7280",
        )
        return

    for area in map_data.get("drivable_areas", {}).values():
        boundary = point_array(area["area_boundary"])
        if len(boundary) >= 3:
            ax.add_patch(
                Polygon(
                    boundary,
                    closed=True,
                    facecolor="#eef1f3",
                    edgecolor="#c7cdd1",
                    linewidth=0.7,
                    zorder=0,
                )
            )

    for crossing in map_data.get("pedestrian_crossings", {}).values():
        edge1 = point_array(crossing["edge1"])
        edge2 = point_array(crossing["edge2"])
        crossing_polygon = np.concatenate([edge1, edge2[::-1]], axis=0)
        ax.add_patch(
            Polygon(
                crossing_polygon,
                closed=True,
                facecolor="#f7e7a9",
                edgecolor="#b99a38",
                linewidth=0.8,
                alpha=0.7,
                zorder=1,
            )
        )

    for lane in map_data.get("lane_segments", {}).values():
        for side in ("left", "right"):
            boundary = point_array(lane[f"{side}_lane_boundary"])
            mark_type = str(lane.get(f"{side}_lane_mark_type", ""))
            linestyle = "--" if "DASHED" in mark_type else "-"
            ax.plot(
                boundary[:, 0],
                boundary[:, 1],
                color="#90989f",
                linewidth=0.75,
                linestyle=linestyle,
                alpha=0.75,
                zorder=2,
            )


def plot_trajectory(
    ax: plt.Axes,
    trajectory: np.ndarray,
    *,
    color: str,
    label: str | None,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 4,
) -> None:
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color=color,
        label=label,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )


def format_probability(value: float) -> str:
    probability = float(value)
    if 0.0 < abs(probability) < 0.001:
        return f"{probability:.2e}"
    return f"{probability:.3f}"


def draw_trajectories(ax: plt.Axes, analysis: dict) -> None:
    top1_mode = analysis["top1_mode"]
    worst_case_mode = analysis["worst_case_mode"]
    other_label_used = False
    for mode_index, trajectory in enumerate(analysis["positions"]):
        if mode_index in (top1_mode, worst_case_mode):
            continue
        plot_trajectory(
            ax,
            trajectory,
            color="#9ca3af",
            label=None if other_label_used else "Other QCNet modes",
            linewidth=1.0,
            alpha=0.55,
            zorder=3,
        )
        other_label_used = True

    plot_trajectory(
        ax,
        analysis["positions"][top1_mode],
        color="#1f4e79",
        label=(
            f"Top-1 QCNet mode {top1_mode} "
            f"(p={format_probability(analysis['probabilities'][top1_mode])})"
        ),
        linewidth=2.5,
        zorder=6,
    )
    plot_trajectory(
        ax,
        analysis["positions"][worst_case_mode],
        color="#b22222",
        label=(
            f"Worst-case QCNet mode {worst_case_mode} "
            f"(p={format_probability(analysis['probabilities'][worst_case_mode])})"
        ),
        linewidth=2.5,
        linestyle="--",
        zorder=6,
    )
    plot_trajectory(
        ax,
        analysis["ego_history"],
        color="#111827",
        label="Ego history",
        linewidth=2.0,
    )
    plot_trajectory(
        ax,
        analysis["ego_future"],
        color="#0891b2",
        label="Ego recorded future",
        linewidth=2.3,
        linestyle="--",
        zorder=5,
    )
    plot_trajectory(
        ax,
        analysis["target_history"],
        color="#166534",
        label="Target history",
        linewidth=2.0,
    )
    plot_trajectory(
        ax,
        analysis["target_future"],
        color="#2e7d32",
        label="Target ground-truth future",
        linewidth=2.3,
        linestyle="-.",
        zorder=5,
    )

    ax.scatter(
        analysis["ego_history"][-1, 0],
        analysis["ego_history"][-1, 1],
        color="#111827",
        marker="o",
        s=42,
        label="Current positions",
        zorder=8,
    )
    ax.scatter(
        analysis["target_history"][-1, 0],
        analysis["target_history"][-1, 1],
        color="#166534",
        marker="o",
        s=42,
        zorder=8,
    )


def mark_minimum_pair(
    ax: plt.Axes,
    ego_future: np.ndarray,
    other_future: np.ndarray,
    timestep: int,
    *,
    color: str,
    marker: str,
    label: str,
) -> None:
    ego_point = ego_future[timestep]
    other_point = other_future[timestep]
    ax.plot(
        [ego_point[0], other_point[0]],
        [ego_point[1], other_point[1]],
        color=color,
        linewidth=1.4,
        linestyle=":",
        zorder=8,
    )
    ax.scatter(
        [ego_point[0], other_point[0]],
        [ego_point[1], other_point[1]],
        color=color,
        marker=marker,
        s=58,
        edgecolors="white",
        linewidths=0.7,
        label=label,
        zorder=9,
    )


def draw_minimum_markers(ax: plt.Axes, analysis: dict) -> None:
    dt = analysis["dt"]
    top1_step = analysis["timestep_of_top1_min"]
    worst_step = analysis["timestep_of_worst_case_min"]
    ground_truth_step = analysis["timestep_of_ground_truth_min"]
    mark_minimum_pair(
        ax,
        analysis["ego_future"],
        analysis["positions"][analysis["top1_mode"]],
        top1_step,
        color="#1f4e79",
        marker="o",
        label=f"Top-1 minimum: step {top1_step}, t={(top1_step + 1) * dt:.1f} s",
    )
    mark_minimum_pair(
        ax,
        analysis["ego_future"],
        analysis["positions"][analysis["worst_case_mode"]],
        worst_step,
        color="#b22222",
        marker="X",
        label=f"Worst-case minimum: step {worst_step}, t={(worst_step + 1) * dt:.1f} s",
    )
    mark_minimum_pair(
        ax,
        analysis["ego_future"],
        analysis["target_future"],
        ground_truth_step,
        color="#2e7d32",
        marker="s",
        label=(
            f"Ground-truth minimum: step {ground_truth_step}, "
            f"t={(ground_truth_step + 1) * dt:.1f} s"
        ),
    )


def trajectory_bounds(analysis: dict) -> tuple[float, float, float, float]:
    arrays = [
        analysis["ego_history"],
        analysis["ego_future"],
        analysis["target_history"],
        analysis["target_future"],
        analysis["positions"].reshape(-1, 2),
    ]
    points = np.concatenate(arrays, axis=0)
    finite = points[np.isfinite(points).all(axis=1)]
    x_min, y_min = np.min(finite, axis=0)
    x_max, y_max = np.max(finite, axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min))
    margin = max(8.0, span * 0.08)
    return x_min - margin, x_max + margin, y_min - margin, y_max + margin


def closest_interaction_center(analysis: dict) -> np.ndarray:
    timestep = analysis["timestep_of_worst_case_min"]
    ego_point = analysis["ego_future"][timestep]
    worst_point = analysis["positions"][analysis["worst_case_mode"], timestep]
    return (ego_point + worst_point) / 2.0


def render_plot(
    metadata: dict,
    analysis: dict,
    map_data: dict | None,
    output_path: Path,
    *,
    zoom_radius: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    draw_map_context(ax, map_data)
    draw_trajectories(ax, analysis)
    draw_minimum_markers(ax, analysis)

    if zoom_radius is None:
        x_min, x_max, y_min, y_max = trajectory_bounds(analysis)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        title_prefix = "Open-loop map and actor context"
    else:
        center = closest_interaction_center(analysis)
        ax.set_xlim(center[0] - zoom_radius, center[0] + zoom_radius)
        ax.set_ylim(center[1] - zoom_radius, center[1] + zoom_radius)
        worst_step = analysis["timestep_of_worst_case_min"]
        title_prefix = (
            "Closest-interaction zoom "
            f"(worst-case step {worst_step}, "
            f"t={(worst_step + 1) * analysis['dt']:.1f} s)"
        )

    map_note = "AV2 map available" if map_data is not None else "AV2 map unavailable"
    ax.set_title(
        f"{title_prefix}: {scenario_label(metadata)}\n"
        f"Scenario {metadata['scenario_id']} | {map_note}"
    )
    ax.set_xlabel("Global x position (m)")
    ax.set_ylabel("Global y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="best", fontsize=7.5, frameon=True, ncol=2)
    fig.text(
        0.5,
        0.015,
        "Open-loop validation using the recorded AV2 ego future; no closed-loop response is simulated.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rounded(value: object) -> float:
    return round(float(value), 6)


def csv_row(metadata: dict, analysis: dict) -> dict:
    top1_mode = analysis["top1_mode"]
    worst_case_mode = analysis["worst_case_mode"]
    return {
        "scenario_id": metadata["scenario_id"],
        "scenario_type": scenario_label(metadata),
        "top1_min_distance": rounded(analysis["top1_min_distance"]),
        "worst_case_min_distance": rounded(analysis["worst_case_min_distance"]),
        "ground_truth_min_distance": rounded(analysis["ground_truth_min_distance"]),
        "timestep_of_top1_min": analysis["timestep_of_top1_min"],
        "timestep_of_worst_case_min": analysis["timestep_of_worst_case_min"],
        "timestep_of_ground_truth_min": analysis["timestep_of_ground_truth_min"],
        "top1_mode": top1_mode,
        "worst_case_mode": worst_case_mode,
        "top1_probability": rounded(analysis["probabilities"][top1_mode]),
        "worst_case_mode_probability": rounded(
            analysis["probabilities"][worst_case_mode]
        ),
    }


def candidate_csv_row(metadata: dict, analysis: dict) -> dict:
    top1_mode = analysis["top1_mode"]
    worst_case_mode = analysis["worst_case_mode"]
    final_timestep = len(analysis["ego_future"]) - 1
    return {
        "scenario_id": metadata["scenario_id"],
        "candidate_type": metadata["candidate_type"],
        "top1_min_distance": rounded(analysis["top1_min_distance"]),
        "worst_case_min_distance": rounded(analysis["worst_case_min_distance"]),
        "ground_truth_min_distance": rounded(analysis["ground_truth_min_distance"]),
        "timestep_of_top1_min": analysis["timestep_of_top1_min"],
        "timestep_of_worst_case_min": analysis["timestep_of_worst_case_min"],
        "timestep_of_ground_truth_min": analysis["timestep_of_ground_truth_min"],
        "top1_mode": top1_mode,
        "worst_case_mode": worst_case_mode,
        "top1_probability": rounded(analysis["probabilities"][top1_mode]),
        "worst_case_mode_probability": rounded(
            analysis["probabilities"][worst_case_mode]
        ),
        "min_occurs_at_horizon_end": (
            "true"
            if analysis["timestep_of_worst_case_min"] == final_timestep
            else "false"
        ),
        "recommended_use": metadata["recommended_use"],
        "notes": metadata["notes"],
    }


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.zoom_radius <= 0:
        raise ValueError("--zoom-radius must be positive")

    ranking = load_ranking(args.ranking_json)
    scenarios = load_scenario_definitions(args.scenario_config)
    candidate_run = args.scenario_config is not None
    rows = []
    for metadata in scenarios:
        scenario_id = metadata["scenario_id"]
        if scenario_id not in ranking:
            raise ValueError(f"Selected scenario missing from ranking: {scenario_id}")
        artifact_path = args.artifact_dir / f"{scenario_id}.npz"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Selected artifact not found: {artifact_path}")

        analysis = analyze_artifact(artifact_path, scenario_id)
        validate_ranking(analysis, ranking[scenario_id])
        map_path = map_path_for_scenario(args.map_root, scenario_id)
        map_data = load_map(map_path)

        overview_path = args.output_dir / f"map_actor_context_{metadata['slug']}.png"
        zoom_path = args.output_dir / f"closest_interaction_{metadata['slug']}.png"
        render_plot(
            metadata,
            analysis,
            map_data,
            overview_path,
            zoom_radius=None,
        )
        render_plot(
            metadata,
            analysis,
            map_data,
            zoom_path,
            zoom_radius=args.zoom_radius,
        )
        rows.append(
            candidate_csv_row(metadata, analysis)
            if candidate_run
            else csv_row(metadata, analysis)
        )
        print(
            f"Created {overview_path} and {zoom_path} "
            f"(map={'yes' if map_data is not None else 'no'})"
        )

    csv_name = args.csv_name or (
        "qcnet_candidate_replacement_validation.csv"
        if candidate_run
        else "qcnet_selected_scenario_context_validation.csv"
    )
    csv_path = args.output_dir / csv_name
    fields = CANDIDATE_CSV_FIELDS if candidate_run else CSV_FIELDS
    write_csv(csv_path, rows, fields)
    print(f"Created {csv_path}")


if __name__ == "__main__":
    main()
