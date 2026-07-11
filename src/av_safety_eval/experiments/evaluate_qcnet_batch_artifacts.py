import argparse
import csv
import json
import os
from glob import glob

import numpy as np


def min_distance(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    distances = np.linalg.norm(a - b, axis=-1)
    distances = distances[valid_mask]
    if distances.size == 0:
        return float("nan")
    return float(np.min(distances))


def evaluate_artifact(path: str, near_miss_threshold: float, collision_threshold: float) -> dict:
    data = np.load(path)

    positions = data["positions"].astype(float)
    probabilities = data["probabilities"].astype(float)
    ego_future = data["ego_future_positions"].astype(float)
    target_future = data["target_future_positions"].astype(float)

    num_modes, horizon_steps, _ = positions.shape
    horizon = min(horizon_steps, ego_future.shape[0], target_future.shape[0])

    positions = positions[:, :horizon]
    ego_future = ego_future[:horizon]
    target_future = target_future[:horizon]

    if "ego_future_valid_mask" in data and "target_future_valid_mask" in data:
        valid_mask = (
            data["ego_future_valid_mask"][:horizon].astype(bool)
            & data["target_future_valid_mask"][:horizon].astype(bool)
        )
    else:
        valid_mask = np.ones(horizon, dtype=bool)

    per_mode_min_distance = []
    for mode_idx in range(num_modes):
        mode_min = min_distance(positions[mode_idx], ego_future, valid_mask)
        per_mode_min_distance.append(mode_min)

    top1_mode = int(np.argmax(probabilities))
    top1_min_distance = float(per_mode_min_distance[top1_mode])
    worst_case_min_distance = float(np.nanmin(per_mode_min_distance))
    ground_truth_min_distance = min_distance(target_future, ego_future, valid_mask)

    multimodal_gap = top1_min_distance - worst_case_min_distance
    prediction_gt_gap = worst_case_min_distance - ground_truth_min_distance

    return {
        "artifact_path": path,
        "scenario_id": str(data["scenario_id"]),
        "target_actor_id": str(data["target_actor_id"]),
        "num_modes": int(num_modes),
        "horizon_steps": int(horizon),
        "dt": float(data["dt"]),
        "probability_sum": float(probabilities.sum()),
        "top1_mode": top1_mode,
        "top1_probability": float(probabilities[top1_mode]),
        "top1_min_distance": top1_min_distance,
        "worst_case_min_distance": worst_case_min_distance,
        "ground_truth_min_distance": float(ground_truth_min_distance),
        "multimodal_gap": float(multimodal_gap),
        "prediction_gt_gap": float(prediction_gt_gap),
        "top1_near_miss": bool(top1_min_distance < near_miss_threshold),
        "multimodal_worst_case_near_miss": bool(worst_case_min_distance < near_miss_threshold),
        "ground_truth_near_miss": bool(ground_truth_min_distance < near_miss_threshold),
        "top1_collision": bool(top1_min_distance < collision_threshold),
        "multimodal_worst_case_collision": bool(worst_case_min_distance < collision_threshold),
        "ground_truth_collision": bool(ground_truth_min_distance < collision_threshold),
        "per_mode_min_distance": [float(x) for x in per_mode_min_distance],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--near-miss-threshold", type=float, default=3.0)
    parser.add_argument("--collision-threshold", type=float, default=1.0)
    args = parser.parse_args()

    paths = sorted(glob(os.path.join(args.artifact_dir, "*.npz")))

    rows = []
    for path in paths:
        try:
            rows.append(evaluate_artifact(path, args.near_miss_threshold, args.collision_threshold))
        except Exception as exc:
            print(f"Skipped {path}: {exc}")

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            row["worst_case_min_distance"],
            row["ground_truth_min_distance"],
            -row["multimodal_gap"],
        ),
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    csv_fields = [
        "scenario_id",
        "target_actor_id",
        "top1_probability",
        "top1_mode",
        "top1_min_distance",
        "worst_case_min_distance",
        "ground_truth_min_distance",
        "multimodal_gap",
        "prediction_gt_gap",
        "top1_near_miss",
        "multimodal_worst_case_near_miss",
        "ground_truth_near_miss",
        "top1_collision",
        "multimodal_worst_case_collision",
        "ground_truth_collision",
        "artifact_path",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({field: row[field] for field in csv_fields})

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(rows_sorted, handle, indent=2)

    print(f"Evaluated {len(rows_sorted)} artifacts")
    print(f"CSV saved to: {args.output_csv}")
    print(f"JSON saved to: {args.output_json}")
    print()
    print("Top 10 riskiest scenarios by worst-case predicted distance:")
    for idx, row in enumerate(rows_sorted[:10], start=1):
        print(
            f"{idx:02d}. {row['scenario_id']} | "
            f"worst={row['worst_case_min_distance']:.3f} m | "
            f"top1={row['top1_min_distance']:.3f} m | "
            f"gt={row['ground_truth_min_distance']:.3f} m | "
            f"gap={row['multimodal_gap']:.3f} m | "
            f"top1_prob={row['top1_probability']:.3f}"
        )


if __name__ == "__main__":
    main()
