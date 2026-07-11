"""Adapter boundary for future cVMD/cVMDx trajectory prediction."""

from __future__ import annotations

from pathlib import Path

from av_safety_eval.common.types import AgentState, PredictionSet
from av_safety_eval.predictors.base import TrajectoryPredictor


class CVMDAdapter(TrajectoryPredictor):
    """Interface placeholder for a future cVMD/cVMDx predictor.

    The adapter is intentionally importable without the external cVMD code,
    PyTorch, highD files, CUDA, or model checkpoints installed. A later
    integration should load the external project lazily and convert model
    samples into this repository's ``PredictionSet`` interface.
    """

    name = "cvmd_adapter"

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.config_path = Path(config_path) if config_path is not None else None

    def predict(
        self,
        history: list[AgentState],
        horizon_steps: int,
        dt: float,
    ) -> PredictionSet:
        """Return multimodal cVMD predictions once the integration exists."""

        raise NotImplementedError(
            "cVMD/cVMDx integration is not implemented yet. "
            "This adapter is a dependency-free boundary for future work; see "
            "docs/cvmd_feasibility_report.md and docs/cvmd_inference_plan.md."
        )
