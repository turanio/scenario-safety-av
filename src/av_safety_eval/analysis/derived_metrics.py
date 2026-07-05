"""Derived metrics computed from per-step experiment logs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _none_if_nan(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def compute_log_derived_metrics(log: pd.DataFrame) -> dict[str, Any]:
    """Compute derived closed-loop metrics from a per-step log DataFrame."""

    required_columns = {
        "step",
        "time",
        "ego_x",
        "ego_vx",
        "action_acceleration",
        "min_distance",
    }
    missing = required_columns - set(log.columns)
    if missing:
        raise ValueError(f"Log is missing required columns: {sorted(missing)}")
    if log.empty:
        raise ValueError("Log must contain at least one row.")

    acceleration = pd.to_numeric(log["action_acceleration"])
    intervention_rows = log[acceleration < 0.0]
    if intervention_rows.empty:
        first_intervention_step = None
        first_intervention_time = None
    else:
        first_intervention = intervention_rows.iloc[0]
        first_intervention_step = int(first_intervention["step"])
        first_intervention_time = float(first_intervention["time"])

    return {
        "first_intervention_step": first_intervention_step,
        "first_intervention_time": first_intervention_time,
        "max_braking": float(acceleration.min()),
        "mean_acceleration": float(acceleration.mean()),
        "average_ego_speed": float(pd.to_numeric(log["ego_vx"]).mean()),
        "final_ego_x": float(pd.to_numeric(log["ego_x"]).iloc[-1]),
        "minimum_logged_distance": float(pd.to_numeric(log["min_distance"]).min()),
    }


def metrics_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert derived metric rows to a stable table-friendly DataFrame."""

    columns = [
        "planner",
        "predictor",
        "first_intervention_step",
        "first_intervention_time",
        "max_braking",
        "mean_acceleration",
        "average_ego_speed",
        "final_ego_x",
        "minimum_logged_distance",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns].map(_none_if_nan)
