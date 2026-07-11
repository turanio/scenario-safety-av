"""Compare SafeIO-style filters on selected QCNet open-loop artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from av_safety_eval.analysis.safety_distances import compute_center_distances
from av_safety_eval.planning.safety_filter import (
    SafetyFilterResult,
    evaluate_probability_aware_filter,
    evaluate_top1_filter,
    evaluate_worst_case_filter,
)


SELECTED_SCENARIOS = (
    {
        "scenario_id": "001749f1-bc1c-47fb-a13f-9ab1f2c050a8",
        "scenario_type": "Hidden risk",
    },
    {
        "scenario_id": "0091bad9-e7b2-4c07-aa12-6b5fd03c63d2",
        "scenario_type": "High-confidence close interaction",
    },
    {
        "scenario_id": "00351569-255c-433e-b97b-e2a844d1b6e0",
        "scenario_type": "Real near-miss",
    },
)

COMPARISON_FIELDS = (
    "scenario_id",
    "scenario_type",
    "policy_name",
    "action",
    "is_safe",
    "trigger_mode",
    "trigger_probability",
    "min_distance",
    "threshold_m",
    "probability_threshold",
    "reason",
)

SUMMARY_FIELDS = (
    "scenario_id",
    "scenario_type",
    "top1_action",
    "worst_case_action",
    "probability_aware_action",
    "top1_trigger_mode",
    "worst_case_trigger_mode",
    "probability_aware_trigger_mode",
    "top1_min_distance",
    "worst_case_min_distance",
    "probability_aware_min_distance",
    "interpretation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("results/qcnet_batch/artifacts"),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("results/qcnet_batch/qcnet_safety_filter_comparison.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/qcnet_batch/qcnet_safety_filter_summary.csv"),
    )
    parser.add_argument("--safety-threshold-m", type=float, default=3.0)
    parser.add_argument("--probability-threshold", type=float, default=0.05)
    return parser.parse_args()


def _valid_mask(data: np.lib.npyio.NpzFile, horizon: int) -> np.ndarray:
    if "ego_future_valid_mask" in data and "target_future_valid_mask" in data:
        return (
            data["ego_future_valid_mask"][:horizon].astype(bool)
            & data["target_future_valid_mask"][:horizon].astype(bool)
        )
    return np.ones(horizon, dtype=bool)


def load_mode_distances(artifact_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load probabilities and aligned per-mode center-distance series."""

    with np.load(artifact_path, allow_pickle=False) as data:
        positions = data["positions"].astype(float)
        probabilities = data["probabilities"].astype(float)
        ego_future = data["ego_future_positions"].astype(float)

        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError(f"Invalid positions shape in artifact: {artifact_path}")
        if ego_future.ndim != 2 or ego_future.shape[1] != 2:
            raise ValueError(f"Invalid ego future shape in artifact: {artifact_path}")

        horizon = min(positions.shape[1], len(ego_future))
        if horizon == 0:
            raise ValueError(f"Empty future horizon in artifact: {artifact_path}")
        mask = _valid_mask(data, horizon)
        if not np.any(mask):
            raise ValueError(f"No valid future steps in artifact: {artifact_path}")

        positions = positions[:, :horizon]
        ego_future = ego_future[:horizon]

    mode_distances = np.stack(
        [
            compute_center_distances(ego_future, mode_positions, mask)
            for mode_positions in positions
        ]
    )
    return probabilities, mode_distances


def rounded(value: float) -> float:
    return round(float(value), 6)


def optional_rounded(value: float | None) -> float | str:
    return "" if value is None else rounded(value)


def optional_mode(value: int | None) -> int | str:
    return "" if value is None else value


def comparison_row(
    metadata: dict,
    result: SafetyFilterResult,
    probability_threshold: float | None,
) -> dict:
    return {
        "scenario_id": metadata["scenario_id"],
        "scenario_type": metadata["scenario_type"],
        "policy_name": result.policy_name,
        "action": result.action,
        "is_safe": "true" if result.is_safe else "false",
        "trigger_mode": optional_mode(result.trigger_mode),
        "trigger_probability": optional_rounded(result.trigger_probability),
        "min_distance": rounded(result.min_distance),
        "threshold_m": rounded(result.threshold_m),
        "probability_threshold": (
            "" if probability_threshold is None else rounded(probability_threshold)
        ),
        "reason": result.reason,
    }


def scenario_interpretation(
    scenario_id: str,
    results: dict[str, SafetyFilterResult],
    probability_threshold: float,
) -> str:
    top1 = results["top1"]
    worst_case = results["worst_case"]
    probability_aware = results["probability_aware"]

    if (
        scenario_id.startswith("001749")
        and top1.action == "NO_BRAKE"
        and worst_case.trigger_mode is not None
        and worst_case.trigger_probability is not None
        and probability_aware.action == "NO_BRAKE"
    ):
        return (
            f"Worst-case filtering brakes for mode {worst_case.trigger_mode} "
            f"(p={worst_case.trigger_probability:.6f}), while top-1 and the "
            f"p >= {probability_threshold:.2f} probability-aware filter do not brake; "
            "the risky mode is below the probability cutoff."
        )
    if scenario_id.startswith("0091bad") and all(
        result.action == "NO_BRAKE" for result in results.values()
    ):
        return (
            "All three filters return NO_BRAKE because their evaluated minima remain "
            f"at or above {top1.threshold_m:.1f} m, consistently identifying a close "
            "interaction without a threshold violation."
        )
    if (
        scenario_id.startswith("003515")
        and all(result.action == "BRAKE" for result in results.values())
        and probability_aware.trigger_mode is not None
        and probability_aware.trigger_probability is not None
    ):
        return (
            f"All three filters brake; the probability-aware trigger is mode "
            f"{probability_aware.trigger_mode} "
            f"(p={probability_aware.trigger_probability:.6f}), which passes the "
            f"p >= {probability_threshold:.2f} cutoff."
        )
    return (
        f"Top-1={top1.action}, worst-case={worst_case.action}, and "
        f"probability-aware={probability_aware.action} under open-loop screening."
    )


def summary_row(
    metadata: dict,
    results: dict[str, SafetyFilterResult],
    probability_threshold: float,
) -> dict:
    top1 = results["top1"]
    worst_case = results["worst_case"]
    probability_aware = results["probability_aware"]
    return {
        "scenario_id": metadata["scenario_id"],
        "scenario_type": metadata["scenario_type"],
        "top1_action": top1.action,
        "worst_case_action": worst_case.action,
        "probability_aware_action": probability_aware.action,
        "top1_trigger_mode": optional_mode(top1.trigger_mode),
        "worst_case_trigger_mode": optional_mode(worst_case.trigger_mode),
        "probability_aware_trigger_mode": optional_mode(
            probability_aware.trigger_mode
        ),
        "top1_min_distance": rounded(top1.min_distance),
        "worst_case_min_distance": rounded(worst_case.min_distance),
        "probability_aware_min_distance": rounded(probability_aware.min_distance),
        "interpretation": scenario_interpretation(
            metadata["scenario_id"], results, probability_threshold
        ),
    }


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    comparison_rows = []
    summary_rows = []

    for metadata in SELECTED_SCENARIOS:
        artifact_path = args.artifact_dir / f"{metadata['scenario_id']}.npz"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Selected artifact not found: {artifact_path}")

        probabilities, mode_distances = load_mode_distances(artifact_path)
        results = {
            "top1": evaluate_top1_filter(
                mode_distances, probabilities, args.safety_threshold_m
            ),
            "worst_case": evaluate_worst_case_filter(
                mode_distances, probabilities, args.safety_threshold_m
            ),
            "probability_aware": evaluate_probability_aware_filter(
                mode_distances,
                probabilities,
                args.safety_threshold_m,
                args.probability_threshold,
            ),
        }

        comparison_rows.extend(
            (
                comparison_row(metadata, results["top1"], None),
                comparison_row(metadata, results["worst_case"], None),
                comparison_row(
                    metadata,
                    results["probability_aware"],
                    args.probability_threshold,
                ),
            )
        )
        summary_rows.append(
            summary_row(metadata, results, args.probability_threshold)
        )

    write_csv(args.comparison_csv, COMPARISON_FIELDS, comparison_rows)
    write_csv(args.summary_csv, SUMMARY_FIELDS, summary_rows)
    print(f"Created {args.comparison_csv}")
    print(f"Created {args.summary_csv}")


if __name__ == "__main__":
    main()
