"""Simulator adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from av_safety_eval.common.types import ControlAction, ScenarioState


class SimulatorAdapter(ABC):
    """Abstract boundary between experiment code and simulator engines."""

    @abstractmethod
    def reset(self) -> ScenarioState:
        """Reset the simulator and return a normalized scenario state."""

    @abstractmethod
    def step(self, action: ControlAction) -> ScenarioState:
        """Apply an action and return a normalized scenario state."""
