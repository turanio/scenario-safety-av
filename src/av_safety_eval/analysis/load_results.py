"""Result loading helpers for analysis scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUMMARY_FILES = {
    "baseline_matrix": Path("results/metrics/baseline_matrix_summary.csv"),
    "closed_loop_baseline": Path("results/metrics/closed_loop_baseline_summary.csv"),
    "planner_comparison": Path("results/metrics/planner_comparison_summary.csv"),
    "uncertainty_comparison": Path("results/metrics/uncertainty_planner_comparison_summary.csv"),
}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file, raising a clear error if it is missing."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_csv_if_exists(path: str | Path) -> pd.DataFrame:
    """Load a CSV file or return an empty frame when absent."""

    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_existing_summaries(project_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load all known summary CSVs that exist under ``project_root``."""

    root = Path(project_root)
    return {
        name: load_csv_if_exists(root / relative_path)
        for name, relative_path in SUMMARY_FILES.items()
    }


def load_logs(project_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load per-step log CSVs from ``results/logs`` keyed by filename stem."""

    logs_dir = Path(project_root) / "results" / "logs"
    if not logs_dir.exists():
        return {}
    return {
        path.stem: pd.read_csv(path)
        for path in sorted(logs_dir.glob("*.csv"))
    }
