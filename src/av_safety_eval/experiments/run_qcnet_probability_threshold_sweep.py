"""Run probability-threshold sensitivity analysis for QCNet multimodal artifacts.

This script evaluates how a probability-aware safety filter changes as the
minimum accepted mode probability changes.

It reads QCNet .npz artifacts and writes:
- per-scenario, per-threshold decisions
- threshold-level summary statistics
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def _as_str(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.tolist())


def _safe_bool_array(data: np.lib.npyio.NpzFile, key: str, length: int) -> np.ndarray:
    if key in data:
        return np.asarray(data[key]).astype(bool)
    return np.ones(length, dtype=bool)


def _pairwise_distances(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid = valid_mask.astype(bool)
    valid &= np.isfinite(a).all(axis=-1)
    valid &= np.isfinite(b).all(axis=-1)

    if not valid.any():
        return np.array([np.inf], dtype=float)

    return np.linalg.norm(a[valid] - b[valid], axis=-1)


def _mode_min_distances(
    predicted_positions: np.ndarray,
    ego_future_positions: np.ndarray,
    ego_future_valid_mask: np.ndarray,
) -> np.ndarray:
    mins = []
    for mode_idx in range(predicted_positions.shape[0]):
        dists = _pairwise_distances(
            predicted_positions[mode_idx],
            ego_future_positions,
            ego_future_valid_mask,
        )
        mins.append(float(np.min(dists)))
    return np.asarray(mins, dtype=float)


def _ground_truth_min_distance(data: np.lib.npyio.NpzFile) -> float:
    if "target_future_positions" not in data or "ego_future_positions" not in data:
        return float("nan")

    target_future = np.asarray(data["target_future_positions"], dtype=float)
    ego_future = np.asarray(data["ego_future_positions"], dtype=float)

    length = min(len(target_future), len(ego_future))
    target_future = target_future[:length]
    ego_future = ego_future[:length]

    ego_mask = _safe_bool_array(data, "ego_future_valid_mask", length)[:length]
    target_mask = _safe_bool_array(data, "target_future_valid_mask", length)[:length]
    valid_mask = ego_mask & target_mask

    dists = _pairwise_distances(target_future, ego_future, valid_mask)
    return float(np.min(dists))


def _iter_artifacts(artifact_dir: Path) -> Iterable[Path]:
    return sorted(p for p in artifact_dir.glob("*.npz") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--per-scenario-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--safety-threshold-m", type=float, default=3.0)
    parser.add_argument(
        "--probability-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50],
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    per_scenario_csv = Path(args.per_scenario_csv)
    summary_csv = Path(args.summary_csv)

    per_scenario_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    artifacts = list(_iter_artifacts(artifact_dir))
    if not artifacts:
        raise FileNotFoundError(f"No .npz artifacts found in {artifact_dir}")

    per_rows = []

    for artifact_path in artifacts:
        data = np.load(artifact_path, allow_pickle=True)

        scenario_id = _as_str(data["scenario_id"])
        target_actor_id = _as_str(data["target_actor_id"]) if "target_actor_id" in data else ""

        positions = np.asarray(data["positions"], dtype=float)
        probabilities = np.asarray(data["probabilities"], dtype=float)
        ego_future = np.asarray(data["ego_future_positions"], dtype=float)

        horizon = min(positions.shape[1], ego_future.shape[0])
        positions = positions[:, :horizon, :]
        ego_future = ego_future[:horizon]

        ego_future_valid_mask = _safe_bool_array(data, "ego_future_valid_mask", horizon)[:horizon]

        mode_min_distances = _mode_min_distances(
            predicted_positions=positions,
            ego_future_positions=ego_future,
            ego_future_valid_mask=ego_future_valid_mask,
        )

        top1_mode = int(np.argmax(probabilities))
        top1_probability = float(probabilities[top1_mode])
        top1_min_distance = float(mode_min_distances[top1_mode])
        top1_brake = top1_min_distance < args.safety_threshold_m

        worst_case_mode = int(np.argmin(mode_min_distances))
        worst_case_probability = float(probabilities[worst_case_mode])
        worst_case_min_distance = float(mode_min_distances[worst_case_mode])
        worst_case_brake = worst_case_min_distance < args.safety_threshold_m

        ground_truth_min_distance = _ground_truth_min_distance(data)
        ground_truth_near_miss = (
            bool(ground_truth_min_distance < args.safety_threshold_m)
            if np.isfinite(ground_truth_min_distance)
            else False
        )

        for threshold in args.probability_thresholds:
            eligible_modes = np.where(probabilities >= threshold)[0]
            fallback_used = False

            if len(eligible_modes) == 0:
                eligible_modes = np.array([top1_mode])
                fallback_used = True

            eligible_distances = mode_min_distances[eligible_modes]
            local_idx = int(np.argmin(eligible_distances))
            trigger_mode = int(eligible_modes[local_idx])
            trigger_probability = float(probabilities[trigger_mode])
            probability_aware_min_distance = float(mode_min_distances[trigger_mode])
            probability_aware_brake = probability_aware_min_distance < args.safety_threshold_m

            per_rows.append(
                {
                    "scenario_id": scenario_id,
                    "target_actor_id": target_actor_id,
                    "probability_threshold": threshold,
                    "eligible_mode_count": int(len(eligible_modes)),
                    "fallback_used": fallback_used,
                    "top1_mode": top1_mode,
                    "top1_probability": top1_probability,
                    "top1_min_distance": top1_min_distance,
                    "top1_brake": top1_brake,
                    "worst_case_mode": worst_case_mode,
                    "worst_case_probability": worst_case_probability,
                    "worst_case_min_distance": worst_case_min_distance,
                    "worst_case_brake": worst_case_brake,
                    "probability_aware_trigger_mode": trigger_mode,
                    "probability_aware_trigger_probability": trigger_probability,
                    "probability_aware_min_distance": probability_aware_min_distance,
                    "probability_aware_brake": probability_aware_brake,
                    "ground_truth_min_distance": ground_truth_min_distance,
                    "ground_truth_near_miss": ground_truth_near_miss,
                    "hidden_risk_detected": (not top1_brake) and probability_aware_brake,
                    "missed_worst_case_brake": worst_case_brake and (not probability_aware_brake),
                    "artifact_path": str(artifact_path),
                }
            )

    per_fieldnames = list(per_rows[0].keys())
    with per_scenario_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_fieldnames)
        writer.writeheader()
        writer.writerows(per_rows)

    summary_rows = []
    thresholds = sorted({float(row["probability_threshold"]) for row in per_rows})

    for threshold in thresholds:
        rows = [row for row in per_rows if float(row["probability_threshold"]) == threshold]
        total = len(rows)

        probability_aware_brake_count = sum(bool(row["probability_aware_brake"]) for row in rows)
        hidden_risk_count = sum(bool(row["hidden_risk_detected"]) for row in rows)
        missed_worst_case_count = sum(bool(row["missed_worst_case_brake"]) for row in rows)
        fallback_count = sum(bool(row["fallback_used"]) for row in rows)

        top1_brake_count = sum(bool(row["top1_brake"]) for row in rows)
        worst_case_brake_count = sum(bool(row["worst_case_brake"]) for row in rows)
        gt_near_miss_count = sum(bool(row["ground_truth_near_miss"]) for row in rows)

        eligible_counts = [int(row["eligible_mode_count"]) for row in rows]
        min_distances = [float(row["probability_aware_min_distance"]) for row in rows]

        summary_rows.append(
            {
                "probability_threshold": threshold,
                "total_scenarios": total,
                "probability_aware_brake_count": probability_aware_brake_count,
                "probability_aware_brake_rate": probability_aware_brake_count / total,
                "top1_brake_count": top1_brake_count,
                "worst_case_brake_count": worst_case_brake_count,
                "ground_truth_near_miss_count": gt_near_miss_count,
                "hidden_risk_detected_count": hidden_risk_count,
                "missed_worst_case_brake_count": missed_worst_case_count,
                "fallback_count": fallback_count,
                "mean_eligible_mode_count": float(np.mean(eligible_counts)),
                "mean_probability_aware_min_distance": float(np.mean(min_distances)),
            }
        )

    summary_fieldnames = list(summary_rows[0].keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Evaluated {len(artifacts)} artifacts")
    print(f"Per-scenario CSV saved to: {per_scenario_csv}")
    print(f"Summary CSV saved to: {summary_csv}")

    print("\nThreshold summary:")
    for row in summary_rows:
        print(
            f"p >= {row['probability_threshold']:.3f} | "
            f"brake={row['probability_aware_brake_count']}/{row['total_scenarios']} | "
            f"hidden={row['hidden_risk_detected_count']} | "
            f"missed_worst={row['missed_worst_case_brake_count']} | "
            f"mean_modes={row['mean_eligible_mode_count']:.2f}"
        )


if __name__ == "__main__":
    main()
