"""Standard deterministic baseline planner."""

from __future__ import annotations

import math

from av_safety_eval.common.types import ControlAction, PredictionSet, ScenarioState
from av_safety_eval.datasets.synthetic import make_constant_velocity_trajectory
from av_safety_eval.metrics.safety import compute_min_distance, compute_time_to_collision
from av_safety_eval.planners.base import Planner


class StandardPlanner(Planner):
    """Simple deterministic planner with distance/TTC based braking."""

    def __init__(
        self,
        near_miss_threshold: float = 3.0,
        collision_threshold: float = 1.0,
        ttc_threshold: float = 2.0,
        gentle_brake: float = -1.0,
        hard_brake: float = -3.0,
    ) -> None:
        if near_miss_threshold <= 0:
            raise ValueError("near_miss_threshold must be positive.")
        if collision_threshold <= 0:
            raise ValueError("collision_threshold must be positive.")
        if ttc_threshold <= 0:
            raise ValueError("ttc_threshold must be positive.")
        self.near_miss_threshold = near_miss_threshold
        self.collision_threshold = collision_threshold
        self.ttc_threshold = ttc_threshold
        self.gentle_brake = gentle_brake
        self.hard_brake = hard_brake

    def plan(self, state: ScenarioState, predictions: list[PredictionSet]) -> ControlAction:
        if not predictions:
            return ControlAction(acceleration=0.0, steering=0.0)

        horizon_steps = min(
            trajectory.steps
            for prediction in predictions
            for trajectory in prediction.trajectories
        )
        ego_future = make_constant_velocity_trajectory(state.ego, horizon_steps, state.dt)
        predicted_min_distance = min(
            compute_min_distance(ego_future, trajectory)
            for prediction in predictions
            for trajectory in prediction.trajectories
        )
        target_state = state.agents[0] if state.agents else None
        time_to_collision = (
            compute_time_to_collision(
                state.ego,
                target_state,
                collision_distance=self.collision_threshold,
                max_time=horizon_steps * state.dt,
            )
            if target_state is not None
            else math.inf
        )

        if predicted_min_distance <= self.collision_threshold or time_to_collision <= self.ttc_threshold:
            return ControlAction(acceleration=self.hard_brake, steering=0.0)
        if predicted_min_distance <= self.near_miss_threshold:
            return ControlAction(acceleration=self.gentle_brake, steering=0.0)
        return ControlAction(acceleration=0.0, steering=0.0)
