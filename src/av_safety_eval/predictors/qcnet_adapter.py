"""Adapter boundary for future QCNet multimodal trajectory prediction."""

from __future__ import annotations

from pathlib import Path

from av_safety_eval.common.types import AgentState, PredictionSet
from av_safety_eval.predictors.base import TrajectoryPredictor


class QCNetAdapter(TrajectoryPredictor):
    """Dependency-free adapter boundary for future QCNet integration.

    The adapter is intentionally importable without QCNet, Argoverse 2,
    PyTorch, PyTorch Geometric, PyTorch Lightning, CARLA, CUDA, or model
    checkpoints installed. A later integration should load QCNet lazily in a
    separate optional environment and convert multimodal predictions into this
    repository's ``PredictionSet`` interface.
    """

    name = "qcnet_adapter"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.config_path = Path(config_path) if config_path is not None else None
        self.device = device

    def predict(
        self,
        history: list[AgentState],
        horizon_steps: int,
        dt: float,
    ) -> PredictionSet:
        """Return QCNet multimodal predictions once the integration exists."""

        raise NotImplementedError(
            "QCNet integration is not implemented yet. "
            "This adapter is a dependency-free boundary for future work; see "
            "docs/qcnet_feasibility_report.md and docs/qcnet_inference_plan.md."
        )
