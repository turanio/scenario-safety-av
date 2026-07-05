"""Constant Velocity baseline predictor."""

from __future__ import annotations

import numpy as np

from av_safety_eval.common.types import AgentState, PredictionSet, Trajectory
from av_safety_eval.predictors.base import TrajectoryPredictor, validate_prediction_inputs


class ConstantVelocityPredictor(TrajectoryPredictor):
    """Predict one future by keeping the latest velocity constant."""

    def predict(
        self,
        history: list[AgentState],
        horizon_steps: int,
        dt: float,
    ) -> PredictionSet:
        latest = validate_prediction_inputs(history, horizon_steps, dt)
        times = np.arange(1, horizon_steps + 1, dtype=float) * dt
        positions = np.column_stack(
            (
                latest.x + latest.vx * times,
                latest.y + latest.vy * times,
            )
        )
        trajectory = Trajectory(agent_id=latest.agent_id, positions=positions, dt=dt)
        return PredictionSet(
            agent_id=latest.agent_id,
            trajectories=[trajectory],
            probabilities=[1.0],
        )
