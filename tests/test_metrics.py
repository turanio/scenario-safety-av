import math

import numpy as np
import pytest

from av_safety_eval.common.types import AgentState, Trajectory
from av_safety_eval.metrics.safety import (
    compute_min_distance,
    compute_time_to_collision,
    is_collision,
    is_near_miss,
)


def test_compute_min_distance() -> None:
    ego = Trajectory(
        agent_id="ego",
        positions=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        dt=1.0,
    )
    other = Trajectory(
        agent_id="target",
        positions=np.array([[3.0, 4.0], [2.0, 0.0], [4.0, 0.0]]),
        dt=1.0,
    )

    assert compute_min_distance(ego, other) == pytest.approx(1.0)


def test_collision_and_near_miss_thresholds() -> None:
    assert is_collision(1.0, threshold=1.0)
    assert not is_collision(1.1, threshold=1.0)
    assert is_near_miss(2.5, threshold=3.0)
    assert not is_near_miss(3.5, threshold=3.0)


def test_time_to_collision_for_closing_agents() -> None:
    ego = AgentState(agent_id="ego", x=0.0, y=0.0, vx=2.0, vy=0.0)
    other = AgentState(agent_id="target", x=5.0, y=0.0, vx=0.0, vy=0.0)

    assert compute_time_to_collision(ego, other, collision_distance=1.0) == pytest.approx(2.0)


def test_time_to_collision_returns_inf_when_not_closing() -> None:
    ego = AgentState(agent_id="ego", x=0.0, y=0.0, vx=0.0, vy=0.0)
    other = AgentState(agent_id="target", x=5.0, y=0.0, vx=1.0, vy=0.0)

    assert math.isinf(compute_time_to_collision(ego, other, collision_distance=1.0))
