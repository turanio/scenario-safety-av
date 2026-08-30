"""Evaluate QCNet risk-score decision policies against recorded AV2 outcomes.

This is offline point-trajectory screening. The QCNet mode weights are not
calibrated collision probabilities, and the recorded AV2 future is one realized
outcome rather than a complete description of safety risk.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from av_safety_eval.experiments.analyze_qcnet_probabilistic_risk import (
    DEFAULT_THRESHOLDS,
    REPRODUCTION_PROFILES,
    analyze_artifact,
    compute_threshold_retention,
    verify_cohort_integrity,
    verify_reproduction,
)


POLICY_FIELDS = (
    "policy_family",
    "policy_name",
    "policy_parameter",
    "parameter_value",
    "total_scenarios",
    "realized_positive_events",
    "total_interventions",
    "intervention_rate",
    "true_positives",
    "false_positives",
    "extra_interventions",
    "false_negatives",
    "true_negatives",
    "realized_event_recall",
    "realized_event_precision",
    "false_positive_rate",
    "f1",
)


def expected_distance_deficit_risk(
    probabilities: Sequence[float],
    mode_min_distances: Sequence[float],
    safety_threshold_m: float = 3.0,
) -> float:
    """Return sum(p_k * max(0, (d_safe - d_k) / d_safe))."""
    if safety_threshold_m <= 0:
        raise ValueError("safety_threshold_m must be positive")
    probability_array = np.asarray(probabilities, dtype=float)
    distance_array = np.asarray(mode_min_distances, dtype=float)
    if probability_array.shape != distance_array.shape:
        raise ValueError("probabilities and mode_min_distances must have matching shapes")
    losses = np.maximum(0.0, (safety_threshold_m - distance_array) / safety_threshold_m)
    return float(np.sum(probability_array * losses))


def probability_filter_decisions(
    records: Sequence[Mapping[str, object]],
    probability_threshold: float,
    safety_threshold_m: float = 3.0,
) -> np.ndarray:
    """Apply the established probability eligibility rule with top-1 fallback."""
    decisions = []
    for row in records:
        probabilities = np.asarray(row["probabilities"], dtype=float)
        distances = np.asarray(row["mode_min_distances"], dtype=float)
        eligible = np.flatnonzero(probabilities >= probability_threshold)
        if eligible.size == 0:
            eligible = np.asarray([int(row["top1_mode"])], dtype=int)
        decisions.append(bool(np.any(distances[eligible] < safety_threshold_m)))
    return np.asarray(decisions, dtype=bool)


def classification_metrics(
    decisions: Sequence[bool],
    realized_events: Sequence[bool],
) -> dict[str, float | int]:
    """Summarize intervention decisions relative to the realized-event reference."""
    predicted = np.asarray(decisions, dtype=bool)
    observed = np.asarray(realized_events, dtype=bool)
    if predicted.shape != observed.shape or predicted.ndim != 1:
        raise ValueError("decisions and realized_events must be matching one-dimensional arrays")
    if predicted.size == 0:
        raise ValueError("decision arrays must not be empty")

    tp = int(np.sum(predicted & observed))
    fp = int(np.sum(predicted & ~observed))
    fn = int(np.sum(~predicted & observed))
    tn = int(np.sum(~predicted & ~observed))
    interventions = int(predicted.sum())
    positives = int(observed.sum())
    recall = tp / positives if positives else 0.0
    precision = tp / interventions if interventions else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total_scenarios": int(predicted.size),
        "realized_positive_events": positives,
        "total_interventions": interventions,
        "intervention_rate": interventions / predicted.size,
        "true_positives": tp,
        "false_positives": fp,
        "extra_interventions": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "realized_event_recall": recall,
        "realized_event_precision": precision,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
    }


def policy_metrics_row(
    family: str,
    name: str,
    parameter: str,
    parameter_value: float | str,
    decisions: Sequence[bool],
    realized_events: Sequence[bool],
) -> dict[str, object]:
    row: dict[str, object] = {
        "policy_family": family,
        "policy_name": name,
        "policy_parameter": parameter,
        "parameter_value": parameter_value,
    }
    row.update(classification_metrics(decisions, realized_events))
    return row


def build_decision_thresholds(scores: Sequence[float]) -> np.ndarray:
    """Combine interpretable fixed points with every observed decision transition."""
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1 or score_array.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(score_array).all() or np.any(score_array < 0):
        raise ValueError("scores must be finite and non-negative")

    fixed = np.round(
        np.concatenate(
            (
                np.asarray(
                    [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 0.001, 0.0025, 0.005]
                ),
                np.arange(0.01, 0.101, 0.01),
                np.arange(0.125, 0.501, 0.025),
                np.arange(0.55, 1.001, 0.05),
            )
        ),
        12,
    )
    observed = np.unique(score_array)
    above_maximum = np.nextafter(float(observed.max()), np.inf)
    return np.unique(np.concatenate((fixed, observed, np.asarray([above_maximum]))))


def sweep_score_policy(
    scores: Sequence[float],
    realized_events: Sequence[bool],
    thresholds: Iterable[float],
    family: str,
    name: str,
    parameter: str,
) -> list[dict[str, object]]:
    score_array = np.asarray(scores, dtype=float)
    return [
        policy_metrics_row(
            family,
            name,
            parameter,
            float(threshold),
            score_array >= float(threshold),
            realized_events,
        )
        for threshold in thresholds
    ]


def binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Compute AUROC as the pairwise ranking probability with half credit for ties."""
    observed = np.asarray(labels, dtype=bool)
    score_array = np.asarray(scores, dtype=float)
    positives = score_array[observed]
    negatives = score_array[~observed]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def binary_auprc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Compute non-interpolated area under the precision-recall curve (average precision)."""
    observed = np.asarray(labels, dtype=bool)
    score_array = np.asarray(scores, dtype=float)
    positive_count = int(observed.sum())
    if positive_count == 0:
        return float("nan")

    order = np.argsort(-score_array, kind="stable")
    sorted_scores = score_array[order]
    sorted_labels = observed[order]
    cumulative_tp = 0
    cumulative_total = 0
    area = 0.0
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        group_positives = int(group.sum())
        cumulative_tp += group_positives
        cumulative_total += int(group.size)
        precision = cumulative_tp / cumulative_total
        area += precision * (group_positives / positive_count)
        start = end
    return float(area)


def reliability_rows(
    labels: Sequence[bool],
    scores: Sequence[float],
    bin_edges: Sequence[float] | None = None,
) -> list[dict[str, object]]:
    """Aggregate unsafe probability mass into fixed-width exploratory bins."""
    observed = np.asarray(labels, dtype=bool)
    score_array = np.asarray(scores, dtype=float)
    edges = np.asarray(bin_edges if bin_edges is not None else np.linspace(0.0, 1.0, 11))
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be a strictly increasing one-dimensional array")

    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (score_array >= lower) & (score_array <= upper)
        else:
            mask = (score_array >= lower) & (score_array < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "scenario_count": count,
                "mean_score": float(np.mean(score_array[mask])) if count else "",
                "observed_event_rate": float(np.mean(observed[mask])) if count else "",
            }
        )
    return rows


def _existing_policy_rows(
    records: Sequence[Mapping[str, object]],
    realized_events: np.ndarray,
    safety_threshold_m: float,
) -> list[dict[str, object]]:
    top1 = np.asarray([row["top1_event"] for row in records], dtype=bool)
    worst = np.asarray([row["worst_case_event"] for row in records], dtype=bool)
    rows = [
        policy_metrics_row("baseline", "top1", "", "", top1, realized_events),
        policy_metrics_row("baseline", "worst_case", "", "", worst, realized_events),
    ]
    for threshold in DEFAULT_THRESHOLDS:
        decisions = probability_filter_decisions(records, threshold, safety_threshold_m)
        rows.append(
            policy_metrics_row(
                "probability_filter",
                f"probability_aware_theta_{threshold:g}",
                "theta",
                float(threshold),
                decisions,
                realized_events,
            )
        )
    return rows


def _per_scenario_row(
    record: Mapping[str, object], safety_threshold_m: float
) -> dict[str, object]:
    probabilities = np.asarray(record["probabilities"], dtype=float)
    distances = np.asarray(record["mode_min_distances"], dtype=float)
    expected_risk = expected_distance_deficit_risk(
        probabilities, distances, safety_threshold_m
    )
    return {
        "scenario_id": record["scenario_id"],
        "target_actor_id": record["target_actor_id"],
        "num_modes": record["num_modes"],
        "top1_mode": record["top1_mode"],
        "top1_probability": record["top1_probability"],
        "top1_min_distance": record["top1_min_distance"],
        "worst_case_mode": record["worst_case_mode"],
        "worst_case_probability": record["worst_case_probability"],
        "worst_case_min_distance": record["worst_case_min_distance"],
        "ground_truth_min_distance": record["ground_truth_min_distance"],
        "unsafe_mode_count": record["unsafe_mode_count"],
        "safety_threshold_m": safety_threshold_m,
        "unsafe_probability_mass": record["unsafe_probability_mass"],
        "severity_weighted_risk": record["severity_weighted_risk"],
        "expected_distance_deficit_risk": expected_risk,
        "top1_event": str(bool(record["top1_event"])).lower(),
        "worst_case_event": str(bool(record["worst_case_event"])).lower(),
        "ground_truth_event": str(bool(record["ground_truth_event"])).lower(),
        "probabilities": json.dumps([float(value) for value in probabilities]),
        "mode_min_distances": json.dumps([float(value) for value in distances]),
    }


def _quality_rows(
    unsafe_mass: np.ndarray,
    expected_risk: np.ndarray,
    realized_events: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metric_rows = [
        {
            "record_type": "score_metric",
            "score_name": "unsafe_probability_mass",
            "metric": "brier_score",
            "value": float(np.mean((unsafe_mass - realized_events.astype(float)) ** 2)),
            "bin_lower": "",
            "bin_upper": "",
            "scenario_count": len(realized_events),
            "mean_score": "",
            "observed_event_rate": "",
            "notes": "Exploratory; QCNet mode weights are not safety-calibrated probabilities.",
        },
        {
            "record_type": "score_metric",
            "score_name": "unsafe_probability_mass",
            "metric": "auroc",
            "value": binary_auroc(realized_events, unsafe_mass),
            "bin_lower": "",
            "bin_upper": "",
            "scenario_count": len(realized_events),
            "mean_score": "",
            "observed_event_rate": "",
            "notes": "Exploratory ranking metric with few realized positive events.",
        },
        {
            "record_type": "score_metric",
            "score_name": "unsafe_probability_mass",
            "metric": "auprc",
            "value": binary_auprc(realized_events, unsafe_mass),
            "bin_lower": "",
            "bin_upper": "",
            "scenario_count": len(realized_events),
            "mean_score": "",
            "observed_event_rate": "",
            "notes": "Non-interpolated average precision; exploratory for this cohort.",
        },
        {
            "record_type": "score_metric",
            "score_name": "expected_distance_deficit_risk",
            "metric": "auroc",
            "value": binary_auroc(realized_events, expected_risk),
            "bin_lower": "",
            "bin_upper": "",
            "scenario_count": len(realized_events),
            "mean_score": "",
            "observed_event_rate": "",
            "notes": "Exploratory ranking metric; this score is not a probability.",
        },
        {
            "record_type": "score_metric",
            "score_name": "expected_distance_deficit_risk",
            "metric": "auprc",
            "value": binary_auprc(realized_events, expected_risk),
            "bin_lower": "",
            "bin_upper": "",
            "scenario_count": len(realized_events),
            "mean_score": "",
            "observed_event_rate": "",
            "notes": "Non-interpolated average precision; exploratory for this cohort.",
        },
    ]
    reliability = reliability_rows(realized_events, unsafe_mass)
    for row in reliability:
        metric_rows.append(
            {
                "record_type": "reliability_bin",
                "score_name": "unsafe_probability_mass",
                "metric": "observed_realized_event_frequency",
                "value": "",
                **row,
                "notes": "Fixed-width bin; empty bins are retained explicitly.",
            }
        )
    return metric_rows, reliability


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_tradeoff(
    path: Path,
    existing_rows: Sequence[Mapping[str, object]],
    mass_rows: Sequence[Mapping[str, object]],
    loss_rows: Sequence[Mapping[str, object]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    families = (
        (existing_rows[2:], "Probability filter", "#176B87", "o"),
        (mass_rows, "Risk mass", "#C73E1D", "s"),
        (loss_rows, "Expected loss", "#6A7F3F", "^"),
    )
    for rows, label, color, marker in families:
        points = sorted(
            {
                (float(row["intervention_rate"]), float(row["realized_event_recall"]))
                for row in rows
                if float(row["intervention_rate"]) <= 0.10
            }
        )
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.8,
            label=label,
        )

    markers = (
        (existing_rows[0], "Top-1", "D"),
        (existing_rows[1], "Worst-case", "X"),
        (
            next(
                row
                for row in existing_rows
                if row["policy_name"] == "probability_aware_theta_0.05"
            ),
            r"Probability-aware $\theta=0.05$",
            "P",
        ),
    )
    for row, label, marker in markers:
        ax.scatter(
            [float(row["intervention_rate"])],
            [float(row["realized_event_recall"])],
            marker=marker,
            s=75,
            edgecolor="black",
            linewidth=0.6,
            label=label,
            zorder=4,
        )

    ax.set_xlabel("Intervention rate")
    ax.set_ylabel("Recall of recorded AV2 threshold events")
    ax.set_title("Exploratory risk-policy trade-off on 500 AV2 scenarios")
    ax.set_xlim(-0.002, 0.102)
    ax.set_ylim(-0.02, 1.04)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(
        0.01,
        0.02,
        "Shown: intervention rate <= 10%; full sweeps are retained in CSV.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_reliability(path: Path, rows: Sequence[Mapping[str, object]], n: int) -> None:
    populated = [row for row in rows if int(row["scenario_count"]) > 0]
    x = np.asarray([row["mean_score"] for row in populated], dtype=float)
    y = np.asarray([row["observed_event_rate"] for row in populated], dtype=float)
    counts = np.asarray([row["scenario_count"] for row in populated], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Identity reference")
    ax.plot(x, y, color="#176B87", linewidth=1.5, alpha=0.8)
    ax.scatter(x, y, s=35 + 2.5 * np.sqrt(counts), color="#C73E1D", zorder=3)
    for x_value, y_value, count in zip(x, y, counts.astype(int)):
        ax.annotate(f"n={count}", (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean unsafe probability mass in bin")
    ax.set_ylabel("Recorded AV2 threshold-event frequency")
    ax.set_title(f"Unsafe-mass reliability (exploratory, n={n})")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _row_by_name(rows: Sequence[Mapping[str, object]], name: str) -> Mapping[str, object]:
    return next(row for row in rows if row["policy_name"] == name)


def _row_at_threshold(
    rows: Sequence[Mapping[str, object]], threshold: float
) -> Mapping[str, object]:
    return min(rows, key=lambda row: abs(float(row["parameter_value"]) - threshold))


def _metrics_table_row(label: str, row: Mapping[str, object]) -> str:
    return (
        f"| {label} | {int(row['total_interventions'])} | "
        f"{100 * float(row['intervention_rate']):.1f}% | {int(row['true_positives'])} | "
        f"{int(row['false_positives'])} | {int(row['false_negatives'])} | "
        f"{float(row['realized_event_recall']):.3f} | "
        f"{float(row['realized_event_precision']):.3f} | "
        f"{float(row['false_positive_rate']):.3f} |"
    )


def _write_summary(
    path: Path,
    records: Sequence[Mapping[str, object]],
    existing_rows: Sequence[Mapping[str, object]],
    mass_rows: Sequence[Mapping[str, object]],
    loss_rows: Sequence[Mapping[str, object]],
    quality_rows: Sequence[Mapping[str, object]],
) -> None:
    quality = {
        (str(row["score_name"]), str(row["metric"])): float(row["value"])
        for row in quality_rows
        if row["record_type"] == "score_metric"
    }
    selected_existing = (
        ("Top-1", _row_by_name(existing_rows, "top1")),
        ("Worst-case", _row_by_name(existing_rows, "worst_case")),
        (
            "Probability-aware theta=0.05",
            _row_by_name(existing_rows, "probability_aware_theta_0.05"),
        ),
    )
    selected_mass = [(rho, _row_at_threshold(mass_rows, rho)) for rho in (0.01, 0.05, 0.10)]
    selected_loss = [(eta, _row_at_threshold(loss_rows, eta)) for eta in (0.01, 0.05, 0.10)]
    realized_count = sum(bool(row["ground_truth_event"]) for row in records)

    lines = [
        "# QCNet Risk-Aware Decision Analysis",
        "",
        "## Scope",
        "",
        f"This Stage C extension uses the existing {len(records)} QCNet artifacts only; QCNet "
        "inference was not rerun. It remains open-loop point-trajectory screening on an AV2 "
        "validation subset, not closed-loop validation or collision-risk estimation.",
        "",
        "## 1. Prediction-distribution risk proxies",
        "",
        "For mode weights `p_k`, mode minimum center distances `d_k`, and `d_safe = 3.0 m`:",
        "",
        "- `M_unsafe = sum p_k I(d_k < d_safe)` is the predicted mode mass in the unsafe screening region.",
        "- `R_expected = sum p_k max(0, (d_safe - d_k) / d_safe)` combines mode mass with normalized threshold deficit.",
        "",
        "Neither score is a calibrated collision probability or physical collision severity.",
        "",
        "## 2. Intervention policies",
        "",
        "The risk-mass policy brakes when `M_unsafe >= rho`; the expected-loss policy brakes "
        "when `R_expected >= eta`. The parameters `rho` and `eta` are operating points that "
        "represent risk tolerance and intervention cost. No threshold is presented as universally optimal.",
        "",
        "## 3. Realized AV2 outcome reference",
        "",
        f"The reference is `ground_truth_min_distance < 3.0 m`, observed in **{realized_count} / "
        f"{len(records)}** scenarios. It describes the one recorded AV2 future, not complete "
        "ground-truth safety risk; non-realized QCNet alternatives may remain plausible.",
        "",
        "### Existing policies",
        "",
        "| Policy | Interventions | Rate | TP | FP / extra | FN | Recall | Precision | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_metrics_table_row(label, row) for label, row in selected_existing)
    lines.extend(
        [
            "",
            "### Selected risk-mass operating points",
            "",
            "| Policy | Interventions | Rate | TP | FP / extra | FN | Recall | Precision | FPR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_metrics_table_row(f"rho={rho:.2f}", row) for rho, row in selected_mass)
    lines.extend(
        [
            "",
            "### Selected expected-loss operating points",
            "",
            "| Policy | Interventions | Rate | TP | FP / extra | FN | Recall | Precision | FPR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_metrics_table_row(f"eta={eta:.2f}", row) for eta, row in selected_loss)
    lines.extend(
        [
            "",
            "The full sweep CSVs include fixed interpretable thresholds and exact observed score "
            "transition values. The curves characterize a conservatism/realized-event-recall "
            "trade-off rather than selecting an optimum.",
            "",
            "## Exploratory score quality",
            "",
            "| Score | Brier | AUROC | AUPRC |",
            "|---|---:|---:|---:|",
            f"| Unsafe probability mass | {quality[('unsafe_probability_mass', 'brier_score')]:.6f} | "
            f"{quality[('unsafe_probability_mass', 'auroc')]:.6f} | "
            f"{quality[('unsafe_probability_mass', 'auprc')]:.6f} |",
            f"| Expected distance-deficit risk | n/a | "
            f"{quality[('expected_distance_deficit_risk', 'auroc')]:.6f} | "
            f"{quality[('expected_distance_deficit_risk', 'auprc')]:.6f} |",
            "",
            "AUPRC is reported as non-interpolated average precision. With only "
            f"{realized_count} realized positives in {len(records)} scenarios, these discrimination "
            "and reliability results are exploratory. No fitting or recalibration was performed, "
            "and they do not support strong calibration claims.",
            "",
            "## Outputs",
            "",
            "- `risk_aware_per_scenario.csv`: direct scores and source point-distance metrics.",
            "- `risk_mass_policy_sweep.csv`: all `rho` operating points.",
            "- `expected_loss_policy_sweep.csv`: all `eta` operating points.",
            "- `existing_policy_realized_outcomes.csv`: top-1, worst-case, and probability-filter results.",
            "- `risk_score_quality.csv`: exploratory score metrics and reliability bins.",
            "- `risk_decision_tradeoff.png`: recall/intervention-rate curves.",
            "- `unsafe_mass_reliability.png`: fixed-bin exploratory reliability view.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("results/qcnet_server_500/artifacts"),
    )
    parser.add_argument(
        "--selected-scenario-ids",
        type=Path,
        default=Path("results/qcnet_server_500/selected_scenario_ids.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qcnet_server_500/risk_aware_decision"),
    )
    parser.add_argument("--safety-threshold-m", type=float, default=3.0)
    parser.add_argument(
        "--reproduction-profile",
        choices=REPRODUCTION_PROFILES,
        default=None,
        help="Use historical_500 to enforce the accepted reproduction fingerprint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_paths = sorted(path for path in args.artifact_dir.glob("*.npz") if path.is_file())
    if not artifact_paths:
        raise FileNotFoundError(f"No QCNet artifacts found in {args.artifact_dir}")
    if not args.selected_scenario_ids.is_file():
        raise FileNotFoundError(f"Missing selected-scenario manifest: {args.selected_scenario_ids}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")

    records = [analyze_artifact(path, args.safety_threshold_m) for path in artifact_paths]
    selected_ids = [
        line.strip()
        for line in args.selected_scenario_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    integrity = verify_cohort_integrity(records, selected_ids)
    if args.reproduction_profile == "historical_500":
        retention = compute_threshold_retention(
            records, DEFAULT_THRESHOLDS, args.safety_threshold_m
        )
        verify_reproduction(records, retention, selected_ids)

    records = sorted(records, key=lambda row: str(row["scenario_id"]))
    per_scenario = [_per_scenario_row(row, args.safety_threshold_m) for row in records]
    realized_events = np.asarray([row["ground_truth_event"] for row in records], dtype=bool)
    unsafe_mass = np.asarray([row["unsafe_probability_mass"] for row in records], dtype=float)
    expected_risk = np.asarray(
        [row["expected_distance_deficit_risk"] for row in per_scenario], dtype=float
    )
    normalized_existing_severity = np.asarray(
        [row["severity_weighted_risk"] for row in records], dtype=float
    ) / args.safety_threshold_m
    if not np.allclose(expected_risk, normalized_existing_severity, atol=1e-12, rtol=0.0):
        raise RuntimeError("Expected-loss scores do not match normalized deficit severity")

    existing_rows = _existing_policy_rows(records, realized_events, args.safety_threshold_m)
    mass_rows = sweep_score_policy(
        unsafe_mass,
        realized_events,
        build_decision_thresholds(unsafe_mass),
        "risk_score",
        "risk_mass",
        "rho",
    )
    loss_rows = sweep_score_policy(
        expected_risk,
        realized_events,
        build_decision_thresholds(expected_risk),
        "risk_score",
        "expected_loss",
        "eta",
    )
    quality_rows, reliability = _quality_rows(unsafe_mass, expected_risk, realized_events)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "risk_aware_per_scenario.csv", per_scenario)
    _write_csv(args.output_dir / "risk_mass_policy_sweep.csv", mass_rows)
    _write_csv(args.output_dir / "expected_loss_policy_sweep.csv", loss_rows)
    _write_csv(args.output_dir / "existing_policy_realized_outcomes.csv", existing_rows)
    _write_csv(args.output_dir / "risk_score_quality.csv", quality_rows)
    _plot_tradeoff(
        args.output_dir / "risk_decision_tradeoff.png",
        existing_rows,
        mass_rows,
        loss_rows,
    )
    _plot_reliability(
        args.output_dir / "unsafe_mass_reliability.png", reliability, len(records)
    )
    _write_summary(
        args.output_dir / "risk_aware_decision_summary.md",
        records,
        existing_rows,
        mass_rows,
        loss_rows,
        quality_rows,
    )

    print("Cohort integrity checks passed:")
    for key, value in integrity.items():
        print(f"  {key}: {value}")
    if args.reproduction_profile == "historical_500":
        print("Historical 500 reproduction fingerprint passed exactly")
    print(f"Realized AV2 threshold events: {int(realized_events.sum())}/{len(records)}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
