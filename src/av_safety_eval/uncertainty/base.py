"""Uncertainty estimator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from av_safety_eval.common.types import PredictionSet


class UncertaintyEstimator(ABC):
    """Abstract interface for deriving uncertainty from prediction samples."""

    @abstractmethod
    def estimate(self, predictions: list[PredictionSet]) -> Any:
        """Estimate uncertainty from prediction sets."""
