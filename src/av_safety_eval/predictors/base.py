"""Predictor interface for future trajectory models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from av_safety_eval.common.types import AgentState, PredictionSet


class TrajectoryPredictor(ABC):
    """Abstract predictor that maps agent history to future trajectories."""

    @abstractmethod
    def predict(
        self,
        history: list[AgentState],
        horizon_steps: int,
        dt: float,
    ) -> PredictionSet:
        """Predict future trajectories for the latest state in ``history``."""


def validate_prediction_inputs(
    history: list[AgentState],
    horizon_steps: int,
    dt: float,
) -> AgentState:
    """Validate common predictor inputs and return the latest state."""

    if not history:
        raise ValueError("history must contain at least one AgentState.")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    return history[-1]
