"""Run the deterministic baseline across a small synthetic scenario matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from av_safety_eval.experiments.baseline_common import (
    BASELINE_SUMMARY_COLUMNS,
    evaluate_constant_velocity_baseline,
    project_root,
    summary_row,
)
from av_safety_eval.scenarios.synthetic_interaction import (
    SyntheticScenarioConfig,
    baseline_matrix_configs,
)


def run_baseline_matrix(
    output_root: str | Path | None = None,
    configs: list[SyntheticScenarioConfig] | None = None,
    make_plots: bool = False,
) -> list[dict[str, Any]]:
    """Run all configured synthetic baseline scenarios and save summary CSV."""

    root = Path(output_root) if output_root is not None else project_root() / "results"
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    scenario_configs = configs if configs is not None else baseline_matrix_configs()
    results = [
        evaluate_constant_velocity_baseline(
            config,
            output_root=root,
            make_plot=make_plots,
        )
        for config in scenario_configs
    ]

    summary_file = metrics_dir / "baseline_matrix_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BASELINE_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_row(result) for result in results)

    return results


def main() -> None:
    """CLI entry point for the baseline matrix."""

    results = run_baseline_matrix()
    print("Baseline matrix complete")
    print(json.dumps([summary_row(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
