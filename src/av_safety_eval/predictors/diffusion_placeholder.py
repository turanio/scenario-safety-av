"""Placeholder for future diffusion-based trajectory predictors."""

from __future__ import annotations

from av_safety_eval.common.types import AgentState, PredictionSet
from av_safety_eval.predictors.base import TrajectoryPredictor


class DiffusionPredictorPlaceholder(TrajectoryPredictor):
    """Explicit boundary for later cVMD/cVMDx integration."""

    def predict(
        self,
        history: list[AgentState],
        horizon_steps: int,
        dt: float,
    ) -> PredictionSet:
        raise NotImplementedError(
            "Diffusion prediction is not implemented in the baseline setup. "
            "Validate cVMD/cVMDx separately before adding this integration."
        )
