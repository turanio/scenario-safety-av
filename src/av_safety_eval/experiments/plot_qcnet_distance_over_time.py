import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SELECTED_SCENARIOS = (
    {
        "scenario_id": "001749f1-bc1c-47fb-a13f-9ab1f2c050a8",
        "scenario_type": "Hidden risk",
        "figure_name": "distance_over_time_hidden_risk_001749.png",
        "interpretation": (
            "Multimodal prediction exposes possible lower-probability risk, "
            "but ground truth remains safe."
        ),
        "planner_interpretation": (
            "The top-1 planner does not brake, while the conservative multimodal "
            "planner brakes for a lower-probability risk; ground truth remains above 3.0 m."
        ),
    },
    {
        "scenario_id": "0058ed53-93bf-42a7-9bba-6df3f6ce20f5",
        "scenario_type": "Large top-1 vs multimodal gap",
        "figure_name": "distance_over_time_large_gap_0058ed.png",
        "interpretation": (
            "Multimodal prediction exposes possible lower-probability risk, "
            "but ground truth remains safe."
        ),
        "planner_interpretation": (
            "The top-1 planner does not brake, while the conservative multimodal "
            "planner brakes for a lower-probability risk; ground truth remains above 3.0 m."
        ),
    },
    {
        "scenario_id": "00351569-255c-433e-b97b-e2a844d1b6e0",
        "scenario_type": "Real near-miss",
        "figure_name": "distance_over_time_real_near_miss_003515.png",
        "interpretation": (
            "Strongest real near-miss example: ground truth, top-1, and worst-case "
            "multimodal distances are all below the 3.0 m near-miss threshold."
        ),
        "planner_interpretation": (
            "Both planners brake, and ground truth also falls below 3.0 m, making "
            "this the strongest real near-miss example in the batch."
        ),
    },
)

SUMMARY_FIELDS = (
    "scenario_id",
    "scenario_type",
    "top1_probability",
    "top1_min_distance",
    "worst_case_min_distance",
    "ground_truth_min_distance",
    "multimodal_gap",
    "interpretation",
)

PLANNER_FIELDS = (
    "scenario_id",
    "scenario_type",
    "top1_min_distance",
    "worst_case_min_distance",
    "ground_truth_min_distance",
    "top1_planner_action",
    "multimodal_planner_action",
    "interpretation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create selected-scenario QCNet distance plots and comparison tables."
    )
    parser.add_argument(
        "--ranking-json",
        type=Path,
        default=Path("results/qcnet_batch/qcnet_batch_scenario_ranking.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("results/qcnet_batch/artifacts"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("results/qcnet_batch/figures"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/qcnet_batch/qcnet_selected_scenarios_summary.csv"),
    )
    parser.add_argument(
        "--planner-csv",
        type=Path,
        default=Path("results/qcnet_batch/qcnet_planner_decision_comparison.csv"),
    )
    parser.add_argument("--near-miss-threshold", type=float, default=3.0)
    parser.add_argument("--collision-threshold", type=float, default=1.0)
    return parser.parse_args()


def load_selected_rows(ranking_path: Path) -> list[tuple[dict, dict]]:
    with ranking_path.open(encoding="utf-8") as handle:
        ranking = json.load(handle)

    if not isinstance(ranking, list):
        raise ValueError(f"Expected a list in ranking JSON: {ranking_path}")

    rows_by_id = {row["scenario_id"]: row for row in ranking}
    selected = []
    for metadata in SELECTED_SCENARIOS:
        scenario_id = metadata["scenario_id"]
        if scenario_id not in rows_by_id:
            raise ValueError(f"Selected scenario missing from ranking JSON: {scenario_id}")
        selected.append((metadata, rows_by_id[scenario_id]))
    return selected


def valid_mask(data: np.lib.npyio.NpzFile, horizon: int) -> np.ndarray:
    if "ego_future_valid_mask" in data and "target_future_valid_mask" in data:
        return (
            data["ego_future_valid_mask"][:horizon].astype(bool)
            & data["target_future_valid_mask"][:horizon].astype(bool)
        )
    return np.ones(horizon, dtype=bool)


def masked_distances(
    trajectory: np.ndarray, ego_future: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    distances = np.linalg.norm(trajectory - ego_future, axis=-1).astype(float)
    distances[~mask] = np.nan
    return distances


def prepare_distance_series(artifact_path: Path) -> dict:
    with np.load(artifact_path, allow_pickle=False) as data:
        positions = data["positions"].astype(float)
        probabilities = data["probabilities"].astype(float)
        ego_future = data["ego_future_positions"].astype(float)
        target_future = data["target_future_positions"].astype(float)
        dt = float(data["dt"])

        horizon = min(positions.shape[1], len(ego_future), len(target_future))
        mask = valid_mask(data, horizon)
        if not np.any(mask):
            raise ValueError(f"No valid future steps in artifact: {artifact_path}")

        positions = positions[:, :horizon]
        ego_future = ego_future[:horizon]
        target_future = target_future[:horizon]

    per_mode_distances = np.stack(
        [masked_distances(mode, ego_future, mask) for mode in positions]
    )
    per_mode_minima = np.nanmin(per_mode_distances, axis=1)
    top1_mode = int(np.argmax(probabilities))
    worst_case_mode = int(np.nanargmin(per_mode_minima))

    return {
        "time": (np.arange(horizon, dtype=float) + 1.0) * dt,
        "probabilities": probabilities,
        "top1_mode": top1_mode,
        "worst_case_mode": worst_case_mode,
        "top1": per_mode_distances[top1_mode],
        "worst_case": per_mode_distances[worst_case_mode],
        "ground_truth": masked_distances(target_future, ego_future, mask),
    }


def validate_against_ranking(series: dict, ranking_row: dict) -> None:
    computed = {
        "top1_min_distance": float(np.nanmin(series["top1"])),
        "worst_case_min_distance": float(np.nanmin(series["worst_case"])),
        "ground_truth_min_distance": float(np.nanmin(series["ground_truth"])),
    }
    for field, value in computed.items():
        expected = float(ranking_row[field])
        if not np.isclose(value, expected, rtol=1e-5, atol=1e-5):
            raise ValueError(
                f"Artifact and ranking disagree for {field}: {value} != {expected}"
            )


def mark_minimum(ax: plt.Axes, time: np.ndarray, distances: np.ndarray, color: str) -> None:
    index = int(np.nanargmin(distances))
    ax.scatter(time[index], distances[index], color=color, s=28, zorder=4)


def plot_scenario(
    metadata: dict,
    ranking_row: dict,
    series: dict,
    output_path: Path,
    near_miss_threshold: float,
    collision_threshold: float,
) -> None:
    time = series["time"]
    top1_color = "#1f4e79"
    worst_color = "#b22222"
    ground_truth_color = "#2e7d32"

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(
        time,
        series["top1"],
        color=top1_color,
        linewidth=2.2,
        label=(
            f"Top-1 QCNet mode {series['top1_mode']} "
            f"(p={float(ranking_row['top1_probability']):.3f})"
        ),
    )
    ax.plot(
        time,
        series["worst_case"],
        color=worst_color,
        linewidth=2.2,
        linestyle="--",
        label=f"Worst-case QCNet mode {series['worst_case_mode']}",
    )
    ax.plot(
        time,
        series["ground_truth"],
        color=ground_truth_color,
        linewidth=2.2,
        linestyle="-.",
        label="Ground-truth target future",
    )

    ax.axhline(
        near_miss_threshold,
        color="#d97706",
        linewidth=1.5,
        linestyle=":",
        label=f"Near-miss threshold ({near_miss_threshold:.1f} m)",
    )
    ax.axhline(
        collision_threshold,
        color="#4b5563",
        linewidth=1.5,
        linestyle=":",
        label=f"Collision threshold ({collision_threshold:.1f} m)",
    )

    mark_minimum(ax, time, series["top1"], top1_color)
    mark_minimum(ax, time, series["worst_case"], worst_color)
    mark_minimum(ax, time, series["ground_truth"], ground_truth_color)

    ax.set_title(
        f"{metadata['scenario_type']}: distance over time\n"
        f"AV2 scenario {metadata['scenario_id']}"
    )
    ax.set_xlabel("Future time (s)")
    ax.set_ylabel("Distance to AV2 ego trajectory (m)")
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8.5, frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def rounded(value: object) -> float:
    return round(float(value), 6)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    selected = load_selected_rows(args.ranking_json)
    summary_rows = []
    planner_rows = []

    for metadata, ranking_row in selected:
        scenario_id = metadata["scenario_id"]
        artifact_path = args.artifact_dir / f"{scenario_id}.npz"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Selected artifact not found: {artifact_path}")

        series = prepare_distance_series(artifact_path)
        validate_against_ranking(series, ranking_row)

        figure_path = args.figure_dir / metadata["figure_name"]
        plot_scenario(
            metadata,
            ranking_row,
            series,
            figure_path,
            args.near_miss_threshold,
            args.collision_threshold,
        )
        print(f"Created {figure_path}")

        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_type": metadata["scenario_type"],
                "top1_probability": rounded(ranking_row["top1_probability"]),
                "top1_min_distance": rounded(ranking_row["top1_min_distance"]),
                "worst_case_min_distance": rounded(ranking_row["worst_case_min_distance"]),
                "ground_truth_min_distance": rounded(ranking_row["ground_truth_min_distance"]),
                "multimodal_gap": rounded(ranking_row["multimodal_gap"]),
                "interpretation": metadata["interpretation"],
            }
        )

        top1_brakes = float(ranking_row["top1_min_distance"]) < args.near_miss_threshold
        multimodal_brakes = (
            float(ranking_row["worst_case_min_distance"]) < args.near_miss_threshold
        )
        planner_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_type": metadata["scenario_type"],
                "top1_min_distance": rounded(ranking_row["top1_min_distance"]),
                "worst_case_min_distance": rounded(ranking_row["worst_case_min_distance"]),
                "ground_truth_min_distance": rounded(ranking_row["ground_truth_min_distance"]),
                "top1_planner_action": "BRAKE" if top1_brakes else "NO_BRAKE",
                "multimodal_planner_action": "BRAKE" if multimodal_brakes else "NO_BRAKE",
                "interpretation": metadata["planner_interpretation"],
            }
        )

    write_csv(args.summary_csv, SUMMARY_FIELDS, summary_rows)
    write_csv(args.planner_csv, PLANNER_FIELDS, planner_rows)
    print(f"Created {args.summary_csv}")
    print(f"Created {args.planner_csv}")


if __name__ == "__main__":
    main()
