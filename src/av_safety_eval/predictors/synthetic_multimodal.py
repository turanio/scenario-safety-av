"""Synthetic multimodal predictor for controlled uncertainty experiments."""

from __future__ import annotations

import numpy as np

from av_safety_eval.common.types import AgentState, PredictionSet, Trajectory
from av_safety_eval.predictors.base import TrajectoryPredictor, validate_prediction_inputs


class SyntheticMultimodalPredictor(TrajectoryPredictor):
    """Return hand-built plausible futures for a target vehicle.

    The modes are deterministic and intentionally simple:

    - keep lane: most likely safe future
    - cut in: lower-probability risky future
    - slow down: low-probability alternative future
    """

    def __init__(
        self,
        probabilities: list[float] | None = None,
        ego_lane_y: float = 0.0,
        cut_in_duration: float = 2.0,
        slow_down_acceleration: float = -1.0,
    ) -> None:
        self.probabilities = probabilities or [0.6, 0.3, 0.1]
        if len(self.probabilities) != 3:
            raise ValueError("SyntheticMultimodalPredictor expects three mode probabilities.")
        if any(probability < 0.0 for probability in self.probabilities):
            raise ValueError("Mode probabilities must be non-negative.")
        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("Mode probabilities must sum to 1.0.")
        if cut_in_duration <= 0.0:
            raise ValueError("cut_in_duration must be positive.")
        self.ego_lane_y = ego_lane_y
        self.cut_in_duration = cut_in_duration
        self.slow_down_acceleration = slow_down_acceleration

    def predict(
        self,
        history: list[AgentState],
        horizon_steps: int,
        dt: float,
    ) -> PredictionSet:
        latest = validate_prediction_inputs(history, horizon_steps, dt)
        times = np.arange(1, horizon_steps + 1, dtype=float) * dt

        keep_lane = np.column_stack(
            (
                latest.x + latest.vx * times,
                np.full_like(times, latest.y),
            )
        )
        cut_in_progress = np.clip(times / self.cut_in_duration, 0.0, 1.0)
        cut_in = np.column_stack(
            (
                latest.x + latest.vx * times,
                latest.y + (self.ego_lane_y - latest.y) * cut_in_progress,
            )
        )
        slow_down_displacement = latest.vx * times + 0.5 * self.slow_down_acceleration * times**2
        slow_down = np.column_stack(
            (
                latest.x + slow_down_displacement,
                np.full_like(times, latest.y),
            )
        )

        trajectories = [
            Trajectory(agent_id=latest.agent_id, positions=keep_lane, dt=dt),
            Trajectory(agent_id=latest.agent_id, positions=cut_in, dt=dt),
            Trajectory(agent_id=latest.agent_id, positions=slow_down, dt=dt),
        ]
        return PredictionSet(
            agent_id=latest.agent_id,
            trajectories=trajectories,
            probabilities=self.probabilities,
        )
