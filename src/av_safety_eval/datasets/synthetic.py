"""Synthetic trajectory helpers used by tests and demos."""

from __future__ import annotations

import numpy as np

from av_safety_eval.common.types import AgentState, Trajectory


def make_constant_velocity_trajectory(
    state: AgentState,
    horizon_steps: int,
    dt: float,
) -> Trajectory:
    """Create a deterministic constant-velocity trajectory from a state."""

    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    times = np.arange(1, horizon_steps + 1, dtype=float) * dt
    positions = np.column_stack((state.x + state.vx * times, state.y + state.vy * times))
    return Trajectory(agent_id=state.agent_id, positions=positions, dt=dt)
