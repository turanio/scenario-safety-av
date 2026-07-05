"""Matplotlib plotting helpers for result summaries."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_MPLCONFIGDIR = Path(os.environ.get("AV_SAFETY_EVAL_MPLCONFIGDIR", "/tmp/av_safety_eval_matplotlib"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from av_safety_eval.analysis.labels import readable_label


def plot_grouped_bar(
    data: pd.DataFrame,
    output_path: str | Path,
    x: str,
    y: str,
    group: str,
    ylabel: str,
    title: str,
    dpi: int = 300,
) -> Path:
    """Save a grouped bar chart from a summary DataFrame."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pivot = data.pivot_table(index=x, columns=group, values=y, aggfunc="first")
    pivot.index = [readable_label(value) for value in pivot.index]
    pivot.columns = [readable_label(value) for value in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title=group.replace("_", " "))
    ax.tick_params(axis="x", rotation=20)
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            if np.isnan(value):
                labels.append("")
            elif float(value).is_integer():
                labels.append(f"{int(value)}")
            else:
                labels.append(f"{value:.2f}")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def plot_log_lines(
    logs_by_label: dict[str, pd.DataFrame],
    output_path: str | Path,
    y_column: str,
    ylabel: str,
    title: str,
    horizontal_lines: list[tuple[float, str, str]] | None = None,
    vertical_lines: list[tuple[float, str, str]] | None = None,
    dpi: int = 300,
) -> Path:
    """Save a line plot from per-step logs keyed by label."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, log in logs_by_label.items():
        ax.plot(log["time"], log[y_column], label=readable_label(label), linewidth=2)
    for y_value, label, color in horizontal_lines or []:
        ax.axhline(y_value, color=color, linestyle="--", linewidth=1.5, label=label)
    for x_value, label, color in vertical_lines or []:
        ax.axvline(x_value, color=color, linestyle=":", linewidth=1.8, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output
