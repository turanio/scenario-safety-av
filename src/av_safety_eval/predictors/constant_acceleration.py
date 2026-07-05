"""Constant Acceleration baseline predictor."""

from __future__ import annotations

import numpy as np

from av_safety_eval.common.types import AgentState, PredictionSet, Trajectory
from av_safety_eval.predictors.base import TrajectoryPredictor, validate_prediction_inputs


class ConstantAccelerationPredictor(TrajectoryPredictor):
    """Predict one future by keeping velocity and acceleration constant."""

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
                latest.x + latest.vx * times + 0.5 * latest.ax * times**2,
                latest.y + latest.vy * times + 0.5 * latest.ay * times**2,
            )
        )
        trajectory = Trajectory(agent_id=latest.agent_id, positions=positions, dt=dt)
        return PredictionSet(
            agent_id=latest.agent_id,
            trajectories=[trajectory],
            probabilities=[1.0],
        )
