"""Evaluate a PredictionSet-compatible QCNet artifact offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from av_safety_eval.metrics.safety import compute_min_distance, is_collision, is_near_miss
from av_safety_eval.predictors.qcnet_output_converter import load_qcnet_npz_prediction


def evaluate_qcnet_artifact(
    artifact_path: str | Path,
    near_miss_threshold: float = 3.0,
    collision_threshold: float = 1.0,
) -> dict[str, Any]:
    """Run a minimal risk check against a documented stationary ego rollout."""

    prediction = load_qcnet_npz_prediction(artifact_path)
    horizon_steps = prediction.trajectories[0].steps
    dt = prediction.trajectories[0].dt
    ego_rollout = np.zeros((horizon_steps, 2), dtype=float)

    per_mode_min_distance = [
        round(compute_min_distance(ego_rollout, trajectory), 6)
        for trajectory in prediction.trajectories
    ]
    worst_case_min_distance = min(per_mode_min_distance)
    probabilities = prediction.probabilities or []

    return {
        "artifact": str(Path(artifact_path)),
        "evaluation_note": "Offline artifact check using a stationary ego rollout at the origin; not a real QCNet thesis result.",
        "target_actor_id": prediction.agent_id,
        "num_modes": len(prediction.trajectories),
        "horizon_steps": horizon_steps,
        "dt": dt,
        "probability_sum": round(float(sum(probabilities)), 6),
        "per_mode_min_distance": per_mode_min_distance,
        "worst_case_min_distance": round(float(worst_case_min_distance), 6),
        "near_miss": is_near_miss(worst_case_min_distance, threshold=near_miss_threshold),
        "collision": is_collision(worst_case_min_distance, threshold=collision_threshold),
        "near_miss_threshold": near_miss_threshold,
        "collision_threshold": collision_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="QCNet .npz artifact path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qcnet_smoke/qcnet_artifact_evaluation.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    result = evaluate_qcnet_artifact(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("QCNet artifact evaluation complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
