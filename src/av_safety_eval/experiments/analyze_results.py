"""Generate thesis-ready analysis tables, plots, and summary text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from av_safety_eval.analysis.derived_metrics import compute_log_derived_metrics, metrics_to_frame
from av_safety_eval.analysis.labels import readable_label
from av_safety_eval.analysis.load_results import load_existing_summaries, load_logs
from av_safety_eval.analysis.plot_results import plot_grouped_bar, plot_log_lines
from av_safety_eval.experiments.baseline_common import project_root

PLANNER_TABLE_COLUMNS = [
    "scenario",
    "planner",
    "predictor",
    "min_distance",
    "near_miss",
    "collision",
    "intervention_count",
    "success",
]

UNCERTAINTY_TABLE_COLUMNS = [
    "scenario",
    "planner",
    "predictor",
    "prediction_modes",
    "min_distance",
    "near_miss",
    "collision",
    "intervention_count",
    "success",
]

KEY_FINDINGS_COLUMNS = [
    "experiment_group",
    "scenario",
    "system",
    "predictor",
    "prediction_modes",
    "min_distance",
    "near_miss",
    "collision",
    "intervention_count",
    "first_intervention_time",
    "interpretation",
]

DELAYED_LOG_SPECS = [
    (
        "standard",
        "constant_velocity",
        "uncertainty_comparison_standard_delayed_cut_in",
    ),
    (
        "uncertainty_aware_conservative",
        "synthetic_multimodal",
        "uncertainty_comparison_uncertainty_aware_conservative_delayed_cut_in",
    ),
]


def _ensure_output_dirs(root: Path) -> dict[str, Path]:
    output_dirs = {
        "figures": root / "results" / "figures",
        "tables": root / "results" / "tables",
        "analysis": root / "results" / "analysis",
        "docs": root / "docs",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return output_dirs


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Summary table is missing required columns: {missing}")
    return frame[columns].copy()


def _save_summary_tables(
    summaries: dict[str, pd.DataFrame],
    logs: dict[str, pd.DataFrame],
    tables_dir: Path,
) -> tuple[dict[str, Path], pd.DataFrame]:
    planner_table = _select_columns(
        summaries.get("planner_comparison", pd.DataFrame()),
        PLANNER_TABLE_COLUMNS,
    )
    uncertainty_table = _select_columns(
        summaries.get("uncertainty_comparison", pd.DataFrame()),
        UNCERTAINTY_TABLE_COLUMNS,
    )

    derived_rows: list[dict[str, Any]] = []
    for planner, predictor, log_key in DELAYED_LOG_SPECS:
        if log_key not in logs:
            continue
        metrics = compute_log_derived_metrics(logs[log_key])
        derived_rows.append({"planner": planner, "predictor": predictor, **metrics})
    delayed_metrics = metrics_to_frame(derived_rows)
    key_findings = _build_key_findings_table(planner_table, uncertainty_table, delayed_metrics)

    table_paths = {
        "planner_comparison_table": tables_dir / "planner_comparison_table.csv",
        "uncertainty_comparison_table": tables_dir / "uncertainty_comparison_table.csv",
        "delayed_cut_in_derived_metrics": tables_dir / "delayed_cut_in_derived_metrics.csv",
        "key_findings_table": tables_dir / "key_findings_table.csv",
    }
    planner_table.to_csv(table_paths["planner_comparison_table"], index=False)
    uncertainty_table.to_csv(table_paths["uncertainty_comparison_table"], index=False)
    delayed_metrics.to_csv(table_paths["delayed_cut_in_derived_metrics"], index=False)
    key_findings.to_csv(table_paths["key_findings_table"], index=False)
    return table_paths, delayed_metrics


def _build_key_findings_table(
    planner_table: pd.DataFrame,
    uncertainty_table: pd.DataFrame,
    delayed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for planner, interpretation in [
        ("naive", "No intervention; collision occurs in the dangerous cut-in case."),
        ("standard", "Reactive braking avoids collision and improves the safety margin."),
    ]:
        result = _lookup_row(planner_table, "collision_risk_cut_in", planner)
        if result is not None:
            rows.append(
                _key_finding_row(
                    experiment_group="Planner comparison",
                    result=result,
                    prediction_modes=None,
                    first_intervention_time=None,
                    interpretation=interpretation,
                )
            )

    delayed_interpretations = {
        "standard": "Single-future planning reacts after the delayed cut-in becomes visible and still produces a near miss.",
        "uncertainty_aware_conservative": "Synthetic multimodal planning considers the plausible cut-in earlier, avoids the near miss, and keeps a larger margin.",
    }
    for planner, interpretation in delayed_interpretations.items():
        result = _lookup_row(uncertainty_table, "delayed_cut_in", planner)
        first_intervention_time = _lookup_metric(delayed_metrics, planner, "first_intervention_time")
        if result is not None:
            rows.append(
                _key_finding_row(
                    experiment_group="Uncertainty comparison",
                    result=result,
                    prediction_modes=result.get("prediction_modes"),
                    first_intervention_time=first_intervention_time,
                    interpretation=interpretation,
                )
            )

    return pd.DataFrame(rows, columns=KEY_FINDINGS_COLUMNS)


def _key_finding_row(
    experiment_group: str,
    result: pd.Series,
    prediction_modes: Any,
    first_intervention_time: Any,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "experiment_group": experiment_group,
        "scenario": readable_label(result["scenario"]),
        "system": readable_label(result["planner"]),
        "predictor": readable_label(result["predictor"]),
        "prediction_modes": "" if prediction_modes is None else str(int(prediction_modes)),
        "min_distance": result["min_distance"],
        "near_miss": result["near_miss"],
        "collision": result["collision"],
        "intervention_count": result["intervention_count"],
        "first_intervention_time": first_intervention_time,
        "interpretation": interpretation,
    }


def _lookup_metric(metrics: pd.DataFrame, planner: str, metric: str) -> Any:
    if metrics.empty or metric not in metrics.columns:
        return None
    matches = metrics[metrics["planner"] == planner]
    if matches.empty:
        return None
    value = matches.iloc[0][metric]
    return None if pd.isna(value) else value


def _save_plots(
    summaries: dict[str, pd.DataFrame],
    logs: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}
    planner_summary = summaries.get("planner_comparison", pd.DataFrame())
    uncertainty_summary = summaries.get("uncertainty_comparison", pd.DataFrame())

    if not planner_summary.empty:
        figure_paths["planner_comparison_min_distance"] = plot_grouped_bar(
            planner_summary,
            figures_dir / "planner_comparison_min_distance.png",
            x="scenario",
            y="min_distance",
            group="planner",
            ylabel="minimum distance (m)",
            title="Planner comparison: minimum distance",
        )
        figure_paths["planner_comparison_interventions"] = plot_grouped_bar(
            planner_summary,
            figures_dir / "planner_comparison_interventions.png",
            x="scenario",
            y="intervention_count",
            group="planner",
            ylabel="intervention count",
            title="Planner comparison: interventions",
        )

    if not uncertainty_summary.empty:
        figure_paths["uncertainty_comparison_min_distance"] = plot_grouped_bar(
            uncertainty_summary,
            figures_dir / "uncertainty_comparison_min_distance.png",
            x="scenario",
            y="min_distance",
            group="planner",
            ylabel="minimum distance (m)",
            title="Uncertainty comparison: minimum distance",
        )

    delayed_logs = {
        planner: logs[log_key]
        for planner, _predictor, log_key in DELAYED_LOG_SPECS
        if log_key in logs
    }
    if delayed_logs:
        figure_paths["delayed_cut_in_distance_over_time"] = plot_log_lines(
            delayed_logs,
            figures_dir / "delayed_cut_in_distance_over_time.png",
            y_column="min_distance",
            ylabel="ego-target distance (m)",
            title="Delayed cut-in: distance over time",
            horizontal_lines=[
                (3.0, "Near-miss threshold (3.0 m)", "tab:orange"),
                (1.0, "Collision threshold (1.0 m)", "tab:red"),
            ],
            vertical_lines=[(1.0, "Cut-in starts (1.0 s)", "black")],
        )
        figure_paths["delayed_cut_in_action_over_time"] = plot_log_lines(
            delayed_logs,
            figures_dir / "delayed_cut_in_action_over_time.png",
            y_column="action_acceleration",
            ylabel="acceleration (m/s²)",
            title="Delayed cut-in: action over time",
            vertical_lines=[(1.0, "Cut-in starts (1.0 s)", "black")],
        )
    return figure_paths


def _lookup_row(frame: pd.DataFrame, scenario: str, planner: str) -> pd.Series | None:
    if frame.empty:
        return None
    matches = frame[(frame["scenario"] == scenario) & (frame["planner"] == planner)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _format_bool(value: Any) -> str:
    return str(bool(value)).lower()


def _write_markdown_summary(
    root: Path,
    docs_dir: Path,
    summaries: dict[str, pd.DataFrame],
    delayed_metrics: pd.DataFrame,
    table_paths: dict[str, Path],
    figure_paths: dict[str, Path],
) -> Path:
    planner_summary = summaries.get("planner_comparison", pd.DataFrame())
    uncertainty_summary = summaries.get("uncertainty_comparison", pd.DataFrame())

    naive_collision = _lookup_row(planner_summary, "collision_risk_cut_in", "naive")
    standard_collision = _lookup_row(planner_summary, "collision_risk_cut_in", "standard")
    delayed_standard = _lookup_row(uncertainty_summary, "delayed_cut_in", "standard")
    delayed_uncertainty = _lookup_row(
        uncertainty_summary,
        "delayed_cut_in",
        "uncertainty_aware_conservative",
    )
    standard_first_intervention = _lookup_metric(delayed_metrics, "standard", "first_intervention_time")
    uncertainty_first_intervention = _lookup_metric(
        delayed_metrics,
        "uncertainty_aware_conservative",
        "first_intervention_time",
    )

    lines = [
        "# Results Summary",
        "",
        "## Available Experiments",
        "",
        "The current repository contains synthetic, CPU-only experiments for open-loop prediction, closed-loop deterministic planning, planner comparison, and synthetic multimodal uncertainty-aware planning.",
        "",
        "No diffusion model, cVMD/cVMDx integration, highD experiment, highway-env experiment, CARLA experiment, or GPU-based result is claimed here.",
        "",
        "## Planner Comparison",
        "",
    ]
    if naive_collision is not None and standard_collision is not None:
        lines.extend(
            [
                "In the `collision_risk_cut_in` scenario, naive planning leads to a collision, while the standard deterministic planner brakes and avoids it.",
                "",
                f"- Naive planner: min distance {naive_collision['min_distance']} m, collision={_format_bool(naive_collision['collision'])}, interventions={naive_collision['intervention_count']}.",
                f"- Standard planner: min distance {standard_collision['min_distance']} m, collision={_format_bool(standard_collision['collision'])}, interventions={standard_collision['intervention_count']}.",
            ]
        )
    else:
        lines.append("Planner comparison results were not available when this summary was generated.")

    lines.extend(["", "## Synthetic Multimodal Uncertainty Comparison", ""])
    if delayed_standard is not None and delayed_uncertainty is not None:
        lines.extend(
            [
                "In the delayed cut-in case, the standard deterministic planner reacts to a single predicted future but still produces a near miss.",
                "The uncertainty-aware conservative planner uses a synthetic multimodal predictor and reacts more conservatively to a plausible lower-probability cut-in future.",
                "",
                f"- Standard planner + Constant Velocity: min distance {delayed_standard['min_distance']} m, near_miss={_format_bool(delayed_standard['near_miss'])}, interventions={delayed_standard['intervention_count']}, first intervention at {standard_first_intervention} s.",
                f"- Uncertainty-aware conservative planner + synthetic multimodal predictor: min distance {delayed_uncertainty['min_distance']} m, near_miss={_format_bool(delayed_uncertainty['near_miss'])}, interventions={delayed_uncertainty['intervention_count']}, first intervention at {uncertainty_first_intervention} s.",
            ]
        )
    else:
        lines.append("Delayed cut-in uncertainty comparison results were not available when this summary was generated.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The results support the thesis logic in a controlled synthetic setting: considering multiple plausible futures can improve safety margins when a lower-probability behavior becomes safety-critical.",
            "The explicit trade-off is conservatism: uncertainty-aware planning avoids the delayed cut-in near miss, but it brakes earlier and intervenes more often.",
            "",
            "## Generated Artifacts",
            "",
            "Tables:",
        ]
    )
    for path in table_paths.values():
        lines.append(f"- `{_relative_path(path, root)}`")
    lines.append("")
    lines.append("Figures:")
    for path in figure_paths.values():
        lines.append(f"- `{_relative_path(path, root)}`")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "These are controlled synthetic experiments. The synthetic multimodal predictor is not a diffusion model, and the uncertainty-aware conservative planner is not a full SafeIO implementation.",
            "Future work should replace or extend the synthetic predictor with a validated diffusion-based predictor and evaluate on more realistic scenarios and datasets.",
        ]
    )

    summary_path = docs_dir / "results_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def run_analysis(project_root_path: str | Path | None = None) -> dict[str, Any]:
    """Run full result analysis and return generated artifact paths."""

    root = Path(project_root_path) if project_root_path is not None else project_root()
    output_dirs = _ensure_output_dirs(root)
    summaries = load_existing_summaries(root)
    logs = load_logs(root)

    table_paths, delayed_metrics = _save_summary_tables(summaries, logs, output_dirs["tables"])
    figure_paths = _save_plots(summaries, logs, output_dirs["figures"])
    markdown_path = _write_markdown_summary(
        root,
        output_dirs["docs"],
        summaries,
        delayed_metrics,
        table_paths,
        figure_paths,
    )

    manifest = {
        "tables": {name: str(path) for name, path in table_paths.items()},
        "figures": {name: str(path) for name, path in figure_paths.items()},
        "summary": str(markdown_path),
    }
    manifest_path = output_dirs["analysis"] / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}


def main() -> None:
    """CLI entry point for thesis result analysis."""

    outputs = run_analysis()
    print("Analysis complete")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
