"""Compare top-1 and multimodal risk from a QCNet artifact offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from av_safety_eval.metrics.safety import compute_min_distance, is_collision, is_near_miss
from av_safety_eval.predictors.qcnet_output_converter import load_qcnet_npz_prediction


def compare_qcnet_artifact_modes(
    artifact_path: str | Path,
    near_miss_threshold: float = 3.0,
    collision_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compare top-1 mode risk with worst-case multimodal risk."""

    artifact_path = Path(artifact_path)
    prediction = load_qcnet_npz_prediction(artifact_path)
    probabilities = prediction.probabilities or [1.0 / len(prediction.trajectories)] * len(
        prediction.trajectories
    )
    top1_mode_index = int(np.argmax(probabilities))
    horizon_steps = prediction.trajectories[0].steps
    ego_rollout = np.zeros((horizon_steps, 2), dtype=float)

    per_mode_min_distance = [
        round(compute_min_distance(ego_rollout, trajectory), 6)
        for trajectory in prediction.trajectories
    ]
    top1_min_distance = per_mode_min_distance[top1_mode_index]
    worst_case_min_distance = min(per_mode_min_distance)

    return {
        "artifact": str(artifact_path),
        "comparison_note": "Offline artifact comparison using a stationary ego rollout; not closed-loop CARLA validation.",
        "fake_artifact_warning": "Fake artifacts are converter tests only and are not real QCNet thesis results.",
        "target_actor_id": prediction.agent_id,
        "num_modes": len(prediction.trajectories),
        "horizon_steps": horizon_steps,
        "dt": prediction.trajectories[0].dt,
        "top1_mode_index": top1_mode_index,
        "top1_probability": round(float(probabilities[top1_mode_index]), 6),
        "top1_min_distance": top1_min_distance,
        "top1_near_miss": is_near_miss(top1_min_distance, threshold=near_miss_threshold),
        "top1_collision": is_collision(top1_min_distance, threshold=collision_threshold),
        "per_mode_min_distance": per_mode_min_distance,
        "worst_case_multimodal_min_distance": round(float(worst_case_min_distance), 6),
        "any_mode_near_miss": any(
            is_near_miss(distance, threshold=near_miss_threshold)
            for distance in per_mode_min_distance
        ),
        "any_mode_collision": any(
            is_collision(distance, threshold=collision_threshold)
            for distance in per_mode_min_distance
        ),
        "near_miss_threshold": near_miss_threshold,
        "collision_threshold": collision_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="QCNet .npz artifact path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qcnet_smoke/qcnet_artifact_planner_comparison.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    result = compare_qcnet_artifact_modes(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("QCNet artifact planner comparison complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
