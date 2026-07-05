"""Basic safety metrics for trajectories and agent states."""

from __future__ import annotations

import math

import numpy as np

from av_safety_eval.common.types import AgentState, Trajectory


def _as_positions(trajectory: Trajectory | np.ndarray) -> np.ndarray:
    if isinstance(trajectory, Trajectory):
        positions = trajectory.positions
    else:
        positions = np.asarray(trajectory, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Trajectory positions must have shape (steps, 2).")
    return positions


def compute_min_distance(
    ego_trajectory: Trajectory | np.ndarray,
    other_trajectory: Trajectory | np.ndarray,
) -> float:
    """Compute minimum Euclidean distance between aligned trajectory steps."""

    ego_positions = _as_positions(ego_trajectory)
    other_positions = _as_positions(other_trajectory)
    steps = min(len(ego_positions), len(other_positions))
    if steps == 0:
        raise ValueError("Trajectories must contain at least one step.")
    distances = np.linalg.norm(ego_positions[:steps] - other_positions[:steps], axis=1)
    return float(distances.min())


def compute_time_to_collision(
    ego_state: AgentState,
    other_state: AgentState,
    collision_distance: float = 1.0,
    max_time: float = 30.0,
) -> float:
    """Estimate time until two constant-velocity discs overlap.

    Returns ``math.inf`` when no collision is predicted within ``max_time``.
    """

    if collision_distance <= 0:
        raise ValueError("collision_distance must be positive.")
    if max_time <= 0:
        raise ValueError("max_time must be positive.")

    relative_position = other_state.position - ego_state.position
    relative_velocity = other_state.velocity - ego_state.velocity
    c = float(np.dot(relative_position, relative_position) - collision_distance**2)
    if c <= 0:
        return 0.0

    a = float(np.dot(relative_velocity, relative_velocity))
    if a == 0.0:
        return math.inf

    b = 2.0 * float(np.dot(relative_position, relative_velocity))
    discriminant = b**2 - 4.0 * a * c
    if discriminant < 0:
        return math.inf

    sqrt_discriminant = math.sqrt(discriminant)
    t_enter = (-b - sqrt_discriminant) / (2.0 * a)
    t_exit = (-b + sqrt_discriminant) / (2.0 * a)
    if t_exit < 0:
        return math.inf

    collision_time = max(0.0, t_enter)
    if collision_time > max_time:
        return math.inf
    return float(collision_time)


def is_collision(distance: float, threshold: float = 1.0) -> bool:
    """Return whether a distance is at or below the collision threshold."""

    if distance < 0:
        raise ValueError("distance must be non-negative.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    return distance <= threshold


def is_near_miss(distance: float, threshold: float = 3.0) -> bool:
    """Return whether a distance is at or below the near-miss threshold."""

    if distance < 0:
        raise ValueError("distance must be non-negative.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    return distance <= threshold
