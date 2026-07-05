"""Small aggregation helpers for experiment outputs."""

from __future__ import annotations

from collections.abc import Iterable

from av_safety_eval.common.types import MetricResult


def metric_results_to_dict(results: Iterable[MetricResult]) -> dict[str, float | int | bool]:
    """Convert metric result objects into a plain dictionary."""

    return {result.name: result.value for result in results}


def boolean_rate(values: Iterable[bool]) -> float:
    """Compute the fraction of true values."""

    value_list = list(values)
    if not value_list:
        raise ValueError("values must not be empty.")
    return sum(bool(value) for value in value_list) / len(value_list)
