"""Select and visualize representative hidden-risk cases from a QCNet batch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from av_safety_eval.experiments.plot_qcnet_distance_over_time import (
    plot_scenario as plot_distance_over_time,
)
from av_safety_eval.experiments.plot_qcnet_distance_over_time import (
    prepare_distance_series,
)
from av_safety_eval.experiments.plot_qcnet_scenario_context_validation import (
    analyze_artifact,
    format_probability,
    load_map,
    map_path_for_scenario,
    render_plot,
)


SUMMARY_FIELDS = (
    "selection_rank",
    "scenario_id",
    "scenario_type",
    "selection_role",
    "recommended_use",
    "top1_action",
    "worst_case_action",
    "top1_min_distance",
    "worst_case_min_distance",
    "ground_truth_min_distance",
    "multimodal_gap",
    "top1_mode",
    "worst_case_mode",
    "top1_probability",
    "worst_case_mode_probability",
    "timestep_of_top1_min",
    "timestep_of_worst_case_min",
    "timestep_of_ground_truth_min",
    "min_occurs_at_horizon_end",
    "ground_truth_below_threshold",
    "map_context_available",
    "selection_reason",
)

VISUAL_NOTES = {
    "001749f1-bc1c-47fb-a13f-9ab1f2c050a8": (
        "The map view shows the interaction around an intersection and makes the "
        "separation between the recorded ego path and target alternatives readable."
    ),
    "032618a4-3f4b-456a-b575-17297fcc1ceb": (
        "The map view shows a lane-following interaction, but the triggering mode has "
        "extremely low probability and occurs late in the horizon."
    ),
    "00e2cd17-25bc-42f2-8f33-17ae24d17a5f": (
        "The map view shows a same-corridor close interaction, with all three minima at "
        "the same early timestep."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select three non-final hidden-risk QCNet cases and create open-loop "
            "map, closest-interaction, and distance-over-time figures."
        )
    )
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=Path(
            "results/qcnet_server_500/qcnet_server_500_ranking.csv"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("results/qcnet_server_500/artifacts"),
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=Path("../data/argoverse2/val/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qcnet_server_500/scenario_validation"),
    )
    parser.add_argument("--near-miss-threshold", type=float, default=3.0)
    parser.add_argument("--meaningful-mode-probability", type=float, default=0.01)
    parser.add_argument("--zoom-radius", type=float, default=15.0)
    parser.add_argument("--ranking-atol", type=float, default=2e-4)
    return parser.parse_args()


def load_ranking(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Ranking CSV is empty: {path}")
    scenario_ids = [row["scenario_id"] for row in rows]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError(f"Ranking CSV contains duplicate scenario IDs: {path}")
    return rows


def is_hidden_risk(ranking_row: dict, threshold_m: float) -> bool:
    top1_action = (
        "BRAKE"
        if float(ranking_row["top1_min_distance"]) < threshold_m
        else "NO_BRAKE"
    )
    worst_case_action = (
        "BRAKE"
        if float(ranking_row["worst_case_min_distance"]) < threshold_m
        else "NO_BRAKE"
    )
    return top1_action == "NO_BRAKE" and worst_case_action == "BRAKE"


def validate_ranking(
    analysis: dict,
    ranking_row: dict,
    *,
    atol: float,
) -> None:
    for field in (
        "top1_min_distance",
        "worst_case_min_distance",
        "ground_truth_min_distance",
    ):
        if not np.isclose(
            float(analysis[field]),
            float(ranking_row[field]),
            rtol=1e-5,
            atol=atol,
        ):
            raise ValueError(
                f"Artifact and ranking disagree for {ranking_row['scenario_id']} "
                f"{field}: {analysis[field]} != {ranking_row[field]}"
            )
    if int(analysis["top1_mode"]) != int(ranking_row["top1_mode"]):
        raise ValueError(
            f"Artifact and ranking disagree on top-1 mode for "
            f"{ranking_row['scenario_id']}"
        )


def build_candidates(
    ranking_rows: list[dict],
    artifact_dir: Path,
    map_root: Path,
    threshold_m: float,
    ranking_atol: float,
) -> list[dict]:
    hidden_rows = [
        row for row in ranking_rows if is_hidden_risk(row, threshold_m)
    ]
    candidates = []
    missing_artifacts = []

    for row in hidden_rows:
        scenario_id = row["scenario_id"]
        artifact_path = artifact_dir / f"{scenario_id}.npz"
        if not artifact_path.is_file():
            missing_artifacts.append(artifact_path)
            continue

        analysis = analyze_artifact(artifact_path, scenario_id)
        validate_ranking(analysis, row, atol=ranking_atol)
        worst_case_mode = int(analysis["worst_case_mode"])
        horizon_end = len(analysis["ego_future"]) - 1
        map_path = map_path_for_scenario(map_root, scenario_id)
        candidates.append(
            {
                "scenario_id": scenario_id,
                "ranking_row": row,
                "artifact_path": artifact_path,
                "analysis": analysis,
                "worst_case_mode_probability": float(
                    analysis["probabilities"][worst_case_mode]
                ),
                "min_occurs_at_horizon_end": (
                    analysis["timestep_of_worst_case_min"] == horizon_end
                ),
                "map_path": map_path,
                "map_context_available": map_path is not None,
            }
        )

    if missing_artifacts:
        preview = "\n".join(str(path) for path in missing_artifacts[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_artifacts)} hidden-risk artifacts. First paths:\n"
            f"{preview}"
        )
    return candidates


def select_hidden_risk_cases(
    candidates: list[dict], meaningful_probability: float
) -> list[dict]:
    non_final = [
        candidate
        for candidate in candidates
        if not candidate["min_occurs_at_horizon_end"]
    ]
    map_ready = [candidate for candidate in non_final if candidate["map_context_available"]]
    pool = map_ready if len(map_ready) >= 3 else non_final
    if len(pool) < 3:
        raise ValueError(
            "Fewer than three hidden-risk candidates have non-final minima; "
            "manual review of final-horizon cases is required."
        )

    tail_case = min(
        pool,
        key=lambda candidate: float(
            candidate["ranking_row"]["worst_case_min_distance"]
        ),
    )
    remaining = [candidate for candidate in pool if candidate is not tail_case]
    meaningful = [
        candidate
        for candidate in remaining
        if candidate["worst_case_mode_probability"] >= meaningful_probability
    ]
    balanced_case = min(
        meaningful or remaining,
        key=lambda candidate: float(
            candidate["ranking_row"]["worst_case_min_distance"]
        ),
    )
    remaining = [candidate for candidate in remaining if candidate is not balanced_case]
    probability_case = max(
        remaining,
        key=lambda candidate: (
            candidate["worst_case_mode_probability"],
            -float(candidate["ranking_row"]["worst_case_min_distance"]),
        ),
    )

    selections = (
        (
            balanced_case,
            "balanced_severity",
            "primary_case_study",
            (
                "Smallest non-final worst-case minimum among candidates whose "
                f"triggering mode has p >= {meaningful_probability:.2f}."
            ),
        ),
        (
            probability_case,
            "higher_probability_threshold_case",
            "secondary_case",
            (
                "Highest worst-case-mode probability among the remaining non-final "
                "hidden-risk candidates."
            ),
        ),
        (
            tail_case,
            "minimum_distance_tail_case",
            "appendix_only",
            (
                "Smallest non-final worst-case point distance; retained to show "
                "extreme-tail sensitivity."
            ),
        ),
    )

    selected = []
    for rank, (candidate, role, recommended_use, reason) in enumerate(
        selections, start=1
    ):
        selected.append(
            {
                **candidate,
                "selection_rank": rank,
                "selection_role": role,
                "recommended_use": recommended_use,
                "selection_reason": reason,
            }
        )
    return selected


def slug_for(scenario_id: str) -> str:
    return f"hidden_risk_{scenario_id[:6]}"


def output_paths(output_dir: Path, selected: list[dict]) -> list[Path]:
    paths = [
        output_dir / "selected_hidden_risk_summary.csv",
        output_dir / "selected_hidden_risk_review.md",
    ]
    for candidate in selected:
        slug = slug_for(candidate["scenario_id"])
        paths.extend(
            [
                output_dir / f"map_actor_context_{slug}.png",
                output_dir / f"closest_interaction_{slug}.png",
                output_dir / f"distance_over_time_{slug}.png",
            ]
        )
    return paths


def assert_outputs_are_new(output_dir: Path, selected: list[dict]) -> None:
    existing = [path for path in output_paths(output_dir, selected) if path.exists()]
    if existing:
        listing = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing scenario-validation outputs:\n" + listing
        )


def rounded(value: object) -> float:
    return round(float(value), 6)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def summary_row(candidate: dict, threshold_m: float) -> dict:
    row = candidate["ranking_row"]
    analysis = candidate["analysis"]
    top1_min = float(row["top1_min_distance"])
    worst_min = float(row["worst_case_min_distance"])
    ground_truth_min = float(row["ground_truth_min_distance"])
    worst_case_mode = int(analysis["worst_case_mode"])
    return {
        "selection_rank": candidate["selection_rank"],
        "scenario_id": candidate["scenario_id"],
        "scenario_type": "Hidden risk",
        "selection_role": candidate["selection_role"],
        "recommended_use": candidate["recommended_use"],
        "top1_action": "BRAKE" if top1_min < threshold_m else "NO_BRAKE",
        "worst_case_action": "BRAKE" if worst_min < threshold_m else "NO_BRAKE",
        "top1_min_distance": rounded(top1_min),
        "worst_case_min_distance": rounded(worst_min),
        "ground_truth_min_distance": rounded(ground_truth_min),
        "multimodal_gap": rounded(row["multimodal_gap"]),
        "top1_mode": int(analysis["top1_mode"]),
        "worst_case_mode": worst_case_mode,
        "top1_probability": rounded(analysis["probabilities"][analysis["top1_mode"]]),
        "worst_case_mode_probability": rounded(
            analysis["probabilities"][worst_case_mode]
        ),
        "timestep_of_top1_min": analysis["timestep_of_top1_min"],
        "timestep_of_worst_case_min": analysis["timestep_of_worst_case_min"],
        "timestep_of_ground_truth_min": analysis["timestep_of_ground_truth_min"],
        "min_occurs_at_horizon_end": bool_text(
            candidate["min_occurs_at_horizon_end"]
        ),
        "ground_truth_below_threshold": bool_text(ground_truth_min < threshold_m),
        "map_context_available": bool_text(candidate["map_context_available"]),
        "selection_reason": candidate["selection_reason"],
    }


def write_summary(path: Path, selected: list[dict], threshold_m: float) -> None:
    rows = [summary_row(candidate, threshold_m) for candidate in selected]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def review_text(
    selected: list[dict],
    total_scenarios: int,
    hidden_risk_count: int,
    threshold_m: float,
    meaningful_probability: float,
) -> str:
    lines = [
        "# Selected Hidden-Risk Scenario Review",
        "",
        "## Selection Basis",
        "",
        (
            f"The {total_scenarios}-scenario QCNet ranking contains "
            f"{hidden_risk_count} hidden-risk cases under the {threshold_m:.1f} m "
            "point-distance threshold. A hidden-risk case has `top1_action == "
            "NO_BRAKE` and `worst_case_action == BRAKE`."
        ),
        "",
        (
            "Cases whose worst-case minimum occurs at the final horizon step were "
            "excluded because at least three non-final alternatives were available. "
            "The three selections represent: the smallest non-final minimum, the "
            f"smallest non-final minimum with triggering-mode p >= "
            f"{meaningful_probability:.2f}, and the highest triggering-mode "
            "probability among the remaining non-final cases. AV2 map context was "
            "available for all three and was visually reviewed."
        ),
        "",
        "## Selected Cases",
        "",
        (
            "| Scenario | Role | Top-1 min (m) | Worst-case min (m) | "
            "Ground truth min (m) | Worst p | Worst step | Use |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in selected:
        row = candidate["ranking_row"]
        analysis = candidate["analysis"]
        lines.append(
            f"| `{candidate['scenario_id']}` | {candidate['selection_role']} | "
            f"{float(row['top1_min_distance']):.3f} | "
            f"{float(row['worst_case_min_distance']):.3f} | "
            f"{float(row['ground_truth_min_distance']):.3f} | "
            f"{format_probability(candidate['worst_case_mode_probability'])} | "
            f"{analysis['timestep_of_worst_case_min']} | "
            f"{candidate['recommended_use']} |"
        )

    lines.extend(["", "## Interpretation", ""])
    for candidate in selected:
        scenario_id = candidate["scenario_id"]
        row = candidate["ranking_row"]
        analysis = candidate["analysis"]
        lines.extend(
            [
                f"### `{scenario_id}`",
                "",
                (
                    f"{candidate['selection_reason']} Top-1 remains at "
                    f"{float(row['top1_min_distance']):.3f} m while mode "
                    f"{analysis['worst_case_mode']} "
                    f"(p={format_probability(candidate['worst_case_mode_probability'])}) "
                    f"reaches {float(row['worst_case_min_distance']):.3f} m at "
                    f"step {analysis['timestep_of_worst_case_min']}. "
                    f"{VISUAL_NOTES.get(scenario_id, 'The AV2 map context is available.')}"
                ),
                "",
                f"[Map and actor context](map_actor_context_{slug_for(scenario_id)}.png) | "
                f"[Closest-interaction zoom](closest_interaction_{slug_for(scenario_id)}.png) | "
                f"[Distance over time](distance_over_time_{slug_for(scenario_id)}.png)",
                "",
            ]
        )

    primary = next(
        candidate
        for candidate in selected
        if candidate["recommended_use"] == "primary_case_study"
    )
    secondary = next(
        candidate
        for candidate in selected
        if candidate["recommended_use"] == "secondary_case"
    )
    appendix = next(
        candidate
        for candidate in selected
        if candidate["recommended_use"] == "appendix_only"
    )
    lines.extend(
        [
            "## Recommendation",
            "",
            (
                f"`{primary['scenario_id']}` is the strongest headline hidden-risk "
                "example. It combines a sub-metre worst-case point distance, a "
                "non-final minimum, a triggering probability above the selection "
                "floor, and readable intersection context. Its ground-truth path "
                "remains above the threshold, so the case demonstrates multimodal "
                "forecast sensitivity rather than a recorded near miss."
            ),
            "",
            (
                f"`{secondary['scenario_id']}` is a useful secondary case because the "
                "worst-case mode has the highest probability in the selected set and "
                "ground truth also falls just below the threshold. It is borderline: "
                "all three minima lie close to 3.0 m, so its classification is "
                "sensitive to the chosen point-distance threshold."
            ),
            "",
            (
                f"`{appendix['scenario_id']}` should be appendix-only. Its very small "
                "worst-case distance is useful for showing worst-case-filter "
                "sensitivity, but the triggering mode has extremely low probability "
                "and occurs late in the forecast horizon."
            ),
            "",
            "## Scope And Caveats",
            "",
            (
                "These figures are open-loop point-trajectory analyses using QCNet "
                "multimodal predictions and the recorded AV2 ego future. They do not "
                "simulate a controller response and are not closed-loop safety proof."
            ),
            "",
            (
                "The distances are between trajectory points, not oriented vehicle "
                "footprints. A value below 1 m or near zero is not evidence of a "
                "confirmed collision. Map context supports qualitative interpretation "
                "but does not replace actor-envelope or vehicle-geometry checking."
            ),
            "",
            (
                "The selected cases are illustrative examples from this 500-scenario "
                "sample. They motivate uncertainty-aware evaluation but do not prove "
                "that a forecasting model or safety filter improves closed-loop safety."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_outputs(
    output_dir: Path,
    selected: list[dict],
    threshold_m: float,
    zoom_radius: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in selected:
        scenario_id = candidate["scenario_id"]
        slug = slug_for(scenario_id)
        metadata = {"scenario_id": scenario_id, "scenario_type": "Hidden risk"}
        analysis = candidate["analysis"]
        map_data = load_map(candidate["map_path"])

        render_plot(
            metadata,
            analysis,
            map_data,
            output_dir / f"map_actor_context_{slug}.png",
            zoom_radius=None,
        )
        render_plot(
            metadata,
            analysis,
            map_data,
            output_dir / f"closest_interaction_{slug}.png",
            zoom_radius=zoom_radius,
        )

        series = prepare_distance_series(candidate["artifact_path"])
        plot_distance_over_time(
            metadata,
            candidate["ranking_row"],
            series,
            output_dir / f"distance_over_time_{slug}.png",
            threshold_m,
            1.0,
        )


def main() -> None:
    args = parse_args()
    if args.near_miss_threshold <= 0:
        raise ValueError("--near-miss-threshold must be positive")
    if not 0 <= args.meaningful_mode_probability <= 1:
        raise ValueError("--meaningful-mode-probability must be in [0, 1]")
    if args.zoom_radius <= 0:
        raise ValueError("--zoom-radius must be positive")

    ranking_rows = load_ranking(args.ranking_csv)
    hidden_risk_count = sum(
        is_hidden_risk(row, args.near_miss_threshold) for row in ranking_rows
    )
    candidates = build_candidates(
        ranking_rows,
        args.artifact_dir,
        args.map_root,
        args.near_miss_threshold,
        args.ranking_atol,
    )
    selected = select_hidden_risk_cases(
        candidates, args.meaningful_mode_probability
    )
    assert_outputs_are_new(args.output_dir, selected)
    render_outputs(
        args.output_dir,
        selected,
        args.near_miss_threshold,
        args.zoom_radius,
    )
    write_summary(
        args.output_dir / "selected_hidden_risk_summary.csv",
        selected,
        args.near_miss_threshold,
    )
    review = review_text(
        selected,
        len(ranking_rows),
        hidden_risk_count,
        args.near_miss_threshold,
        args.meaningful_mode_probability,
    )
    review_path = args.output_dir / "selected_hidden_risk_review.md"
    with review_path.open("x", encoding="utf-8") as handle:
        handle.write(review)

    for candidate in selected:
        print(
            f"Selected {candidate['scenario_id']} "
            f"({candidate['selection_role']})"
        )
    for path in output_paths(args.output_dir, selected):
        print(f"Created {path}")


if __name__ == "__main__":
    main()
