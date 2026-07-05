"""Conservative uncertainty-aware planner for sampled future trajectories."""

from __future__ import annotations

from av_safety_eval.common.types import ControlAction, PredictionSet, ScenarioState
from av_safety_eval.datasets.synthetic import make_constant_velocity_trajectory
from av_safety_eval.metrics.safety import compute_min_distance
from av_safety_eval.planners.base import Planner


class ConservativeUncertaintyPlanner(Planner):
    """SafeIO-style conservative planner over sampled trajectory futures.

    This class currently uses a simple sampled-trajectory distance check. It is
    intentionally not a full conformal, diffusion, or SafeIO implementation.
    """

    def __init__(
        self,
        safety_distance: float = 3.0,
        braking_acceleration: float = -2.0,
    ) -> None:
        if safety_distance <= 0:
            raise ValueError("safety_distance must be positive.")
        self.safety_distance = safety_distance
        self.braking_acceleration = braking_acceleration

    def plan(self, state: ScenarioState, predictions: list[PredictionSet]) -> ControlAction:
        if not predictions:
            return ControlAction(acceleration=0.0, steering=0.0)

        horizon_steps = min(
            trajectory.steps
            for prediction in predictions
            for trajectory in prediction.trajectories
        )
        ego_future = make_constant_velocity_trajectory(state.ego, horizon_steps, state.dt)
        risky = any(
            compute_min_distance(ego_future, trajectory) <= self.safety_distance
            for prediction in predictions
            for trajectory_index, trajectory in enumerate(prediction.trajectories)
            if prediction.probabilities is None or prediction.probabilities[trajectory_index] > 0.0
        )
        if risky:
            return ControlAction(acceleration=self.braking_acceleration, steering=0.0)
        return ControlAction(acceleration=0.0, steering=0.0)
