import argparse
import json
import os

import numpy as np


def min_distance(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    distances = np.linalg.norm(a - b, axis=-1)
    distances = distances[valid_mask]
    if distances.size == 0:
        return float("nan")
    return float(np.min(distances))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--near-miss-threshold", type=float, default=3.0)
    parser.add_argument("--collision-threshold", type=float, default=1.0)
    args = parser.parse_args()

    data = np.load(args.artifact)

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
        valid_mask = data["ego_future_valid_mask"][:horizon].astype(bool) & data["target_future_valid_mask"][:horizon].astype(bool)
    else:
        valid_mask = np.ones(horizon, dtype=bool)

    per_mode_min_distance = []
    for mode_idx in range(num_modes):
        mode_min = min_distance(positions[mode_idx], ego_future, valid_mask)
        per_mode_min_distance.append(round(mode_min, 6))

    top1_mode = int(np.argmax(probabilities))
    top1_min_distance = float(per_mode_min_distance[top1_mode])
    worst_case_min_distance = float(np.nanmin(per_mode_min_distance))
    probability_weighted_min_distance = float(np.sum(probabilities * np.asarray(per_mode_min_distance)))

    ground_truth_min_distance = min_distance(target_future, ego_future, valid_mask)

    result = {
        "artifact": args.artifact,
        "evaluation_note": (
            "Open-loop scenario check using real AV2 ego future trajectory and QCNet multimodal "
            "focal-agent predictions. This is not yet a closed-loop planner result."
        ),
        "scenario_id": str(data["scenario_id"]),
        "target_actor_id": str(data["target_actor_id"]),
        "ego_actor_id": str(data["ego_actor_id"]) if "ego_actor_id" in data else "AV",
        "coordinate_frame": str(data["coordinate_frame"]),
        "source": str(data["source"]),
        "num_modes": int(num_modes),
        "horizon_steps": int(horizon),
        "dt": float(data["dt"]),
        "probability_sum": float(probabilities.sum()),
        "probabilities": [round(float(x), 6) for x in probabilities],
        "top1_mode": top1_mode,
        "top1_probability": round(float(probabilities[top1_mode]), 6),
        "per_mode_min_distance": per_mode_min_distance,
        "top1_min_distance": round(top1_min_distance, 6),
        "worst_case_min_distance": round(worst_case_min_distance, 6),
        "probability_weighted_min_distance": round(probability_weighted_min_distance, 6),
        "ground_truth_min_distance": round(float(ground_truth_min_distance), 6),
        "top1_near_miss": bool(top1_min_distance < args.near_miss_threshold),
        "multimodal_worst_case_near_miss": bool(worst_case_min_distance < args.near_miss_threshold),
        "ground_truth_near_miss": bool(ground_truth_min_distance < args.near_miss_threshold),
        "top1_collision": bool(top1_min_distance < args.collision_threshold),
        "multimodal_worst_case_collision": bool(worst_case_min_distance < args.collision_threshold),
        "ground_truth_collision": bool(ground_truth_min_distance < args.collision_threshold),
        "near_miss_threshold": args.near_miss_threshold,
        "collision_threshold": args.collision_threshold,
    }

    print("QCNet scenario artifact evaluation complete")
    print(json.dumps(result, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Saved evaluation JSON to: {args.output}")


if __name__ == "__main__":
    main()
