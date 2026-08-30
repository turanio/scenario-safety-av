"""Compute probability-weighted QCNet safety-risk proxies from exported artifacts.

The analysis uses center-to-center point-trajectory distances over jointly valid
ego and focal-target future timesteps. The probability terms are QCNet mode
weights, not calibrated collision probabilities, and the distance deficit is a
screening proxy rather than physical collision severity.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_THRESHOLDS = (0.0, 0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50)
EXPECTED_COUNTS = {
    "total_scenarios": 500,
    "worst_case_events": 31,
    "top1_events": 13,
    "ground_truth_events": 8,
    "hidden_risk_cases": 18,
}
EXPECTED_SWEEP = {
    0.000: (31, 18, 0, 6.000, 0),
    0.001: (29, 16, 2, 5.462, 0),
    0.010: (24, 11, 7, 4.950, 0),
    0.030: (20, 7, 11, 4.508, 0),
    0.050: (19, 6, 12, 4.138, 0),
    0.100: (14, 1, 17, 3.424, 0),
    0.200: (14, 1, 17, 2.246, 0),
    0.300: (13, 0, 18, 1.256, 110),
    0.500: (13, 0, 18, 1.000, 392),
}
KEY_SCENARIOS = {
    "001749": {
        "scenario_id": "001749f1-bc1c-47fb-a13f-9ab1f2c050a8",
        "top1_min_distance": 3.317130489,
        "worst_case_min_distance": 0.448097372,
        "ground_truth_min_distance": 3.406408448,
        "worst_case_probability": 0.024045080,
    },
    "00e2cd": {
        "scenario_id": "00e2cd17-25bc-42f2-8f33-17ae24d17a5f",
        "top1_min_distance": 3.037804988,
        "worst_case_min_distance": 2.867814728,
        "ground_truth_min_distance": 2.941304898,
        "worst_case_probability": 0.141277462,
    },
    "003515": {
        "scenario_id": "00351569-255c-433e-b97b-e2a844d1b6e0",
        "top1_min_distance": 2.596660757,
        "worst_case_min_distance": 2.044681911,
        "ground_truth_min_distance": 2.161665887,
    },
    "032618": {
        "scenario_id": "032618a4-3f4b-456a-b575-17297fcc1ceb",
        "top1_min_distance": 5.503256101,
        "worst_case_min_distance": 0.272483961,
        "ground_truth_min_distance": 5.322972208,
        "worst_case_probability": 0.000007162456,
    },
}


def _as_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(array.tolist())


def _valid_mask(data: np.lib.npyio.NpzFile, key: str, horizon: int) -> np.ndarray:
    if key not in data:
        return np.ones(horizon, dtype=bool)
    return np.asarray(data[key][:horizon], dtype=bool)


def _minimum_distance(a: np.ndarray, b: np.ndarray, valid_mask: np.ndarray) -> float:
    finite = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    usable = np.asarray(valid_mask, dtype=bool) & finite
    if not usable.any():
        return float("inf")
    return float(np.linalg.norm(a[usable] - b[usable], axis=-1).min())


def analyze_artifact(path: Path, safety_threshold_m: float = 3.0) -> Dict[str, object]:
    """Return per-scenario point-distance and probability-weighted risk proxies."""
    with np.load(path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"], dtype=float)
        probabilities = np.asarray(data["probabilities"], dtype=float).reshape(-1)
        ego_future = np.asarray(data["ego_future_positions"], dtype=float)
        target_future = np.asarray(data["target_future_positions"], dtype=float)

        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError(f"{path}: positions must have shape [modes, steps, 2]")
        if probabilities.shape[0] != positions.shape[0]:
            raise ValueError(f"{path}: probability and mode counts differ")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError(f"{path}: probabilities must be finite and non-negative")

        horizon = min(positions.shape[1], ego_future.shape[0], target_future.shape[0])
        if horizon <= 0:
            raise ValueError(f"{path}: artifact has no common future horizon")

        positions = positions[:, :horizon]
        ego_future = ego_future[:horizon]
        target_future = target_future[:horizon]
        joint_mask = (
            _valid_mask(data, "ego_future_valid_mask", horizon)
            & _valid_mask(data, "target_future_valid_mask", horizon)
        )
        if not joint_mask.any():
            raise ValueError(f"{path}: artifact has no jointly valid future timestep")

        mode_min_distances = np.asarray(
            [_minimum_distance(mode, ego_future, joint_mask) for mode in positions],
            dtype=float,
        )
        ground_truth_min_distance = _minimum_distance(target_future, ego_future, joint_mask)

        top1_mode = int(np.argmax(probabilities))
        worst_case_mode = int(np.argmin(mode_min_distances))
        unsafe_modes = mode_min_distances < safety_threshold_m
        deficits = np.maximum(0.0, safety_threshold_m - mode_min_distances)

        top1_min_distance = float(mode_min_distances[top1_mode])
        worst_case_min_distance = float(mode_min_distances[worst_case_mode])

        return {
            "scenario_id": _as_text(data["scenario_id"]),
            "target_actor_id": _as_text(data["target_actor_id"]),
            "num_modes": int(positions.shape[0]),
            "top1_mode": top1_mode,
            "top1_probability": float(probabilities[top1_mode]),
            "top1_min_distance": top1_min_distance,
            "worst_case_mode": worst_case_mode,
            "worst_case_probability": float(probabilities[worst_case_mode]),
            "worst_case_min_distance": worst_case_min_distance,
            "ground_truth_min_distance": float(ground_truth_min_distance),
            "multimodal_gap": top1_min_distance - worst_case_min_distance,
            "unsafe_mode_count": int(unsafe_modes.sum()),
            "unsafe_probability_mass": float(probabilities[unsafe_modes].sum()),
            "severity_weighted_risk": float(np.sum(probabilities * deficits)),
            "top1_event": bool(top1_min_distance < safety_threshold_m),
            "worst_case_event": bool(worst_case_min_distance < safety_threshold_m),
            "ground_truth_event": bool(ground_truth_min_distance < safety_threshold_m),
            "mode_min_distances": mode_min_distances,
            "probabilities": probabilities,
        }


def compute_threshold_retention(
    records: Sequence[Mapping[str, object]],
    probability_thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    safety_threshold_m: float = 3.0,
) -> List[Dict[str, object]]:
    """Aggregate retained probability and severity with top-1 fallback."""
    full_mass = float(sum(float(row["unsafe_probability_mass"]) for row in records))
    full_severity = float(sum(float(row["severity_weighted_risk"]) for row in records))
    output = []

    for threshold in probability_thresholds:
        triggered = 0
        hidden = 0
        missed_worst = 0
        fallback_count = 0
        eligible_counts = []
        retained_masses = []
        retained_unsafe_mass = 0.0
        retained_severity = 0.0

        for row in records:
            probabilities = np.asarray(row["probabilities"], dtype=float)
            distances = np.asarray(row["mode_min_distances"], dtype=float)
            eligible = np.flatnonzero(probabilities >= threshold)
            if eligible.size == 0:
                eligible = np.asarray([int(row["top1_mode"])], dtype=int)
                fallback_count += 1

            selected_distances = distances[eligible]
            selected_probabilities = probabilities[eligible]
            selected_unsafe = selected_distances < safety_threshold_m
            event = bool(selected_unsafe.any())

            triggered += int(event)
            hidden += int(event and not bool(row["top1_event"]))
            missed_worst += int(bool(row["worst_case_event"]) and not event)
            eligible_counts.append(int(eligible.size))
            retained_masses.append(float(selected_probabilities.sum()))
            retained_unsafe_mass += float(selected_probabilities[selected_unsafe].sum())
            retained_severity += float(
                np.sum(
                    selected_probabilities
                    * np.maximum(0.0, safety_threshold_m - selected_distances)
                )
            )

        output.append(
            {
                "probability_threshold": float(threshold),
                "total_scenarios": len(records),
                "triggered_scenarios": triggered,
                "hidden_risk_detected_count": hidden,
                "missed_worst_case_count": missed_worst,
                "fallback_scenarios": fallback_count,
                "mean_eligible_modes": float(np.mean(eligible_counts)),
                "mean_retained_probability_mass": float(np.mean(retained_masses)),
                "total_unsafe_probability_mass_retained": retained_unsafe_mass,
                "full_distribution_unsafe_probability_mass": full_mass,
                "unsafe_probability_mass_retained_pct": (
                    100.0 * retained_unsafe_mass / full_mass if full_mass > 0 else 100.0
                ),
                "total_probability_weighted_severity_retained": retained_severity,
                "full_distribution_probability_weighted_severity": full_severity,
                "probability_weighted_severity_retained_pct": (
                    100.0 * retained_severity / full_severity if full_severity > 0 else 100.0
                ),
            }
        )

    return output


def verify_reproduction(
    records: Sequence[Mapping[str, object]],
    threshold_rows: Sequence[Mapping[str, object]],
    selected_scenario_ids: Sequence[str],
    distance_tolerance: float = 5e-4,
    probability_tolerance: float = 1e-5,
) -> Dict[str, int]:
    """Fail fast unless the recreated artifacts match the validated experiment."""
    counts = {
        "total_scenarios": len(records),
        "worst_case_events": sum(bool(row["worst_case_event"]) for row in records),
        "top1_events": sum(bool(row["top1_event"]) for row in records),
        "ground_truth_events": sum(bool(row["ground_truth_event"]) for row in records),
        "hidden_risk_cases": sum(
            bool(row["worst_case_event"]) and not bool(row["top1_event"])
            for row in records
        ),
    }
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Reproduction count mismatch: expected {EXPECTED_COUNTS}, got {counts}")

    artifact_ids = {str(row["scenario_id"]) for row in records}
    selected_ids = set(selected_scenario_ids)
    if len(selected_scenario_ids) != 500 or len(selected_ids) != 500:
        raise RuntimeError("Selected scenario manifest must contain exactly 500 unique IDs")
    if artifact_ids != selected_ids:
        missing = sorted(selected_ids - artifact_ids)[:5]
        extra = sorted(artifact_ids - selected_ids)[:5]
        raise RuntimeError(f"Artifact/manifest ID mismatch: missing={missing}, extra={extra}")

    sweep_by_threshold = {float(row["probability_threshold"]): row for row in threshold_rows}
    for threshold, expected in EXPECTED_SWEEP.items():
        if threshold not in sweep_by_threshold:
            raise RuntimeError(f"Missing probability threshold {threshold:.3f}")
        row = sweep_by_threshold[threshold]
        observed = (
            int(row["triggered_scenarios"]),
            int(row["hidden_risk_detected_count"]),
            int(row["missed_worst_case_count"]),
            float(row["mean_eligible_modes"]),
            int(row["fallback_scenarios"]),
        )
        if observed[:3] != expected[:3] or observed[4] != expected[4]:
            raise RuntimeError(
                f"Threshold {threshold:.3f} count mismatch: expected {expected}, got {observed}"
            )
        if not np.isclose(observed[3], expected[3], atol=1e-9, rtol=0.0):
            raise RuntimeError(
                f"Threshold {threshold:.3f} mean-mode mismatch: "
                f"expected {expected[3]}, got {observed[3]}"
            )

    by_id = {str(row["scenario_id"]): row for row in records}
    for label, expected in KEY_SCENARIOS.items():
        scenario_id = str(expected["scenario_id"])
        if scenario_id not in by_id:
            raise RuntimeError(f"Missing key reproduction scenario {label}: {scenario_id}")
        observed = by_id[scenario_id]
        for metric in (
            "top1_min_distance",
            "worst_case_min_distance",
            "ground_truth_min_distance",
            "worst_case_probability",
        ):
            if metric not in expected:
                continue
            tolerance = (
                probability_tolerance if metric.endswith("probability") else distance_tolerance
            )
            if not np.isclose(
                float(observed[metric]), float(expected[metric]), atol=tolerance, rtol=0.0
            ):
                raise RuntimeError(
                    f"Scenario {label} {metric} mismatch: "
                    f"expected {expected[metric]}, got {observed[metric]}"
                )

    return counts


def _public_row(record: Mapping[str, object]) -> Dict[str, object]:
    fields = (
        "scenario_id",
        "target_actor_id",
        "num_modes",
        "top1_mode",
        "top1_probability",
        "top1_min_distance",
        "worst_case_mode",
        "worst_case_probability",
        "worst_case_min_distance",
        "ground_truth_min_distance",
        "multimodal_gap",
        "unsafe_mode_count",
        "unsafe_probability_mass",
        "severity_weighted_risk",
        "top1_event",
        "worst_case_event",
        "ground_truth_event",
    )
    row = {field: record[field] for field in fields}
    row["mode_min_distances"] = json.dumps(
        [float(value) for value in np.asarray(record["mode_min_distances"], dtype=float)]
    )
    return row


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {name: float("nan") for name in ("mean", "median", "p25", "p75", "p90", "p95")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def _distribution_table(all_values: Sequence[float]) -> List[Tuple[str, Dict[str, float]]]:
    positive = [value for value in all_values if value > 0]
    return [("All 500 scenarios", _distribution(all_values)), ("Risk-positive only", _distribution(positive))]


def _format_float(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def _write_summary(
    path: Path,
    records: Sequence[Mapping[str, object]],
    threshold_rows: Sequence[Mapping[str, object]],
    counts: Mapping[str, int],
) -> None:
    mass_values = [float(row["unsafe_probability_mass"]) for row in records]
    severity_values = [float(row["severity_weighted_risk"]) for row in records]
    risk_positive = sum(value > 0 for value in mass_values)
    top_rows = sorted(
        records,
        key=lambda row: (
            -float(row["severity_weighted_risk"]),
            -float(row["unsafe_probability_mass"]),
            str(row["scenario_id"]),
        ),
    )[:15]
    by_id = {str(row["scenario_id"]): row for row in records}

    lines = [
        "# QCNet Probabilistic-Risk Proxy Analysis",
        "",
        "## Scope and definitions",
        "",
        "This report post-processes the reproduced 500-scenario QCNet/Argoverse 2 validation subset. "
        "For each predicted mode, the minimum center-to-center ego/target distance is computed over "
        "jointly valid future timesteps using the 3.0 m screening threshold.",
        "",
        "`unsafe_probability_mass` is the sum of QCNet mode weights whose minimum distance is below "
        "3.0 m. `severity_weighted_risk` is the probability-weighted distance deficit below 3.0 m. "
        "Both are risk proxies: QCNet probabilities are not safety-calibrated collision probabilities, "
        "and the deficit is not physical expected collision severity.",
        "",
        "## Reproduction sanity checks",
        "",
        "All mandatory checks passed before these Stage C outputs were written.",
        "",
        "| Check | Reproduced | Required |",
        "|---|---:|---:|",
        f"| Scenarios | {counts['total_scenarios']} | 500 |",
        f"| Worst-case threshold events | {counts['worst_case_events']} | 31 |",
        f"| Top-1 threshold events | {counts['top1_events']} | 13 |",
        f"| Recorded-ground-truth threshold events | {counts['ground_truth_events']} | 8 |",
        f"| Hidden-risk cases | {counts['hidden_risk_cases']} | 18 |",
        "",
        f"Risk-positive scenarios: **{risk_positive} / {len(records)}**.",
        "",
    ]

    for heading, values in (
        ("Unsafe probability mass", mass_values),
        ("Probability-weighted distance-deficit severity", severity_values),
    ):
        lines.extend(
            [
                f"## {heading}",
                "",
                "| Population | Mean | Median | P25 | P75 | P90 | P95 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for population, stats in _distribution_table(values):
            lines.append(
                f"| {population} | {stats['mean']:.6f} | {stats['median']:.6f} | "
                f"{stats['p25']:.6f} | {stats['p75']:.6f} | {stats['p90']:.6f} | "
                f"{stats['p95']:.6f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Probability-threshold retention",
            "",
            "When no mode reaches a probability threshold, the highest-probability mode is retained "
            "as the same top-1 fallback used by the existing probability-aware filter.",
            "",
            "| Threshold | Triggered | Fallback | Mean modes | Mean retained mass | Unsafe mass retained | Severity retained |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_rows:
        lines.append(
            f"| {float(row['probability_threshold']):.3f} | {int(row['triggered_scenarios'])} | "
            f"{int(row['fallback_scenarios'])} | {float(row['mean_eligible_modes']):.3f} | "
            f"{float(row['mean_retained_probability_mass']):.6f} | "
            f"{float(row['unsafe_probability_mass_retained_pct']):.2f}% | "
            f"{float(row['probability_weighted_severity_retained_pct']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Top 15 by probability-weighted severity",
            "",
            "| Rank | Scenario | Unsafe modes | Unsafe mass | Severity proxy | Top-1 min (m) | Worst min (m) | GT min (m) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_rows, start=1):
        lines.append(
            f"| {rank} | `{row['scenario_id']}` | {int(row['unsafe_mode_count'])} | "
            f"{_format_float(row['unsafe_probability_mass'])} | "
            f"{_format_float(row['severity_weighted_risk'])} | "
            f"{float(row['top1_min_distance']):.3f} | "
            f"{float(row['worst_case_min_distance']):.3f} | "
            f"{float(row['ground_truth_min_distance']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Key scenario checks",
            "",
            "| Scenario | Top-1 probability | Worst-mode probability | Top-1 min (m) | Worst min (m) | GT min (m) | Unsafe mass | Severity proxy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("001749", "00e2cd", "032618"):
        row = by_id[str(KEY_SCENARIOS[label]["scenario_id"])]
        lines.append(
            f"| `{label}` | {_format_float(row['top1_probability'])} | "
            f"{_format_float(row['worst_case_probability'], 9)} | "
            f"{float(row['top1_min_distance']):.3f} | "
            f"{float(row['worst_case_min_distance']):.3f} | "
            f"{float(row['ground_truth_min_distance']):.3f} | "
            f"{_format_float(row['unsafe_probability_mass'], 9)} | "
            f"{_format_float(row['severity_weighted_risk'], 9)} |"
        )

    lines.extend(
        [
            "",
            "## Factual interpretation",
            "",
            "The two proxies distinguish low-probability severe alternatives from cases where more "
            "probability mass lies below the point-distance threshold. Probability filtering reduces "
            "the retained proxy totals as the cutoff rises, quantifying the trade-off already visible "
            "in the policy event counts. This remains open-loop point-trajectory screening; it does not "
            "establish calibrated collision risk, exact vehicle overlap, collision avoidance, or "
            "closed-loop safety improvement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_scatter(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    mass = np.asarray([row["unsafe_probability_mass"] for row in records], dtype=float)
    severity = np.asarray([row["severity_weighted_risk"] for row in records], dtype=float)
    by_id = {str(row["scenario_id"]): row for row in records}

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter(mass, severity, s=25, alpha=0.65, color="#176B87", edgecolors="none")
    label_offsets = {"001749": (12, 18), "00e2cd": (10, 12), "032618": (12, -20)}
    for label in ("001749", "00e2cd", "032618"):
        row = by_id[str(KEY_SCENARIOS[label]["scenario_id"])]
        x = float(row["unsafe_probability_mass"])
        y = float(row["severity_weighted_risk"])
        ax.scatter([x], [y], s=54, color="#C73E1D", zorder=3)
        ax.annotate(
            label,
            (x, y),
            xytext=label_offsets[label],
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
        )
    ax.set_xlabel("Unsafe probability mass (risk proxy)")
    ax.set_ylabel("Probability-weighted distance deficit (m)")
    ax.set_title("QCNet probabilistic-risk proxies across 500 AV2 scenarios")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_threshold_retention(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    thresholds = [float(row["probability_threshold"]) for row in rows]
    positions = np.arange(len(thresholds))
    unsafe = [float(row["unsafe_probability_mass_retained_pct"]) for row in rows]
    severity = [float(row["probability_weighted_severity_retained_pct"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(positions, unsafe, marker="o", linewidth=2, label="Unsafe probability mass")
    ax.plot(positions, severity, marker="s", linewidth=2, label="Weighted distance deficit")
    ax.set_xlabel("Minimum retained QCNet mode probability")
    ax.set_ylabel("Full-distribution proxy retained (%)")
    ax.set_title("Risk-proxy retention under probability filtering")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{threshold:.3f}" for threshold in thresholds], rotation=35, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default="results/qcnet_server_500/artifacts",
        type=Path,
    )
    parser.add_argument(
        "--selected-scenario-ids",
        default="results/qcnet_server_500/selected_scenario_ids.txt",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="results/qcnet_server_500/probabilistic_risk",
        type=Path,
    )
    parser.add_argument("--safety-threshold-m", default=3.0, type=float)
    parser.add_argument(
        "--probability-thresholds",
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        type=float,
    )
    args = parser.parse_args()

    artifact_paths = sorted(path for path in args.artifact_dir.glob("*.npz") if path.is_file())
    if not artifact_paths:
        raise FileNotFoundError(f"No QCNet artifacts found in {args.artifact_dir}")
    if not args.selected_scenario_ids.is_file():
        raise FileNotFoundError(f"Missing selected-scenario manifest: {args.selected_scenario_ids}")

    records = [analyze_artifact(path, args.safety_threshold_m) for path in artifact_paths]
    threshold_rows = compute_threshold_retention(
        records,
        probability_thresholds=args.probability_thresholds,
        safety_threshold_m=args.safety_threshold_m,
    )
    selected_ids = [
        line.strip()
        for line in args.selected_scenario_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = verify_reproduction(records, threshold_rows, selected_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_scenario_path = args.output_dir / "probabilistic_risk_per_scenario.csv"
    threshold_path = args.output_dir / "probabilistic_risk_threshold_summary.csv"
    summary_path = args.output_dir / "probabilistic_risk_summary.md"
    scatter_path = args.output_dir / "probabilistic_risk_scatter.png"
    retention_path = args.output_dir / "probabilistic_risk_threshold_retention.png"

    public_rows = [_public_row(row) for row in sorted(records, key=lambda row: str(row["scenario_id"]))]
    _write_csv(per_scenario_path, public_rows)
    _write_csv(threshold_path, threshold_rows)
    _write_summary(summary_path, records, threshold_rows, counts)
    _plot_scatter(scatter_path, records)
    _plot_threshold_retention(retention_path, threshold_rows)

    print("Mandatory reproduction checks passed exactly:")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    print(f"Analyzed {len(records)} artifacts")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
