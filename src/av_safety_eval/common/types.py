"""Core dataclasses shared across the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AgentState:
    """State of one traffic participant at one time step."""

    agent_id: str
    x: float
    y: float
    vx: float
    vy: float
    ax: float = 0.0
    ay: float = 0.0
    heading: float = 0.0

    @property
    def position(self) -> np.ndarray:
        """Return the two-dimensional position as ``[x, y]``."""

        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        """Return the two-dimensional velocity as ``[vx, vy]``."""

        return np.array([self.vx, self.vy], dtype=float)

    @property
    def acceleration(self) -> np.ndarray:
        """Return the two-dimensional acceleration as ``[ax, ay]``."""

        return np.array([self.ax, self.ay], dtype=float)


@dataclass
class Trajectory:
    """Future or observed positions for one agent.

    ``positions`` must have shape ``(steps, 2)`` with columns ``x`` and ``y``.
    """

    agent_id: str
    positions: np.ndarray
    dt: float

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("Trajectory positions must have shape (steps, 2).")
        if self.dt <= 0:
            raise ValueError("Trajectory dt must be positive.")
        self.positions = positions

    @property
    def steps(self) -> int:
        """Number of trajectory steps."""

        return int(self.positions.shape[0])


@dataclass
class PredictionSet:
    """One or more predicted futures for one agent."""

    agent_id: str
    trajectories: list[Trajectory]
    probabilities: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("PredictionSet must contain at least one trajectory.")
        if self.probabilities is not None:
            if len(self.probabilities) != len(self.trajectories):
                raise ValueError("Prediction probabilities must match trajectories.")
            if any(probability < 0 for probability in self.probabilities):
                raise ValueError("Prediction probabilities must be non-negative.")


@dataclass
class ControlAction:
    """Simple longitudinal/lateral control command."""

    acceleration: float
    steering: float = 0.0


@dataclass
class ScenarioState:
    """Current state returned by scenarios and simulators."""

    ego: AgentState
    agents: list[AgentState]
    time_step: int = 0
    dt: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_seconds(self) -> float:
        """Current scenario time in seconds."""

        return self.time_step * self.dt


@dataclass
class MetricResult:
    """Named metric value with optional unit and metadata."""

    name: str
    value: float | int | bool
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
