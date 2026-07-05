"""Simple envelope over sampled predicted trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from av_safety_eval.common.types import PredictionSet
from av_safety_eval.uncertainty.base import UncertaintyEstimator


@dataclass
class PredictionEnvelope:
    """Per-step min/max bounds for predicted positions."""

    agent_id: str
    lower: np.ndarray
    upper: np.ndarray


class SampledEnvelopeEstimator(UncertaintyEstimator):
    """Build axis-aligned envelopes from sampled trajectory predictions."""

    def estimate(self, predictions: list[PredictionSet]) -> dict[str, PredictionEnvelope]:
        envelopes: dict[str, PredictionEnvelope] = {}
        for prediction in predictions:
            samples = np.stack([trajectory.positions for trajectory in prediction.trajectories])
            envelopes[prediction.agent_id] = PredictionEnvelope(
                agent_id=prediction.agent_id,
                lower=samples.min(axis=0),
                upper=samples.max(axis=0),
            )
        return envelopes
