"""Scenario interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from av_safety_eval.common.types import ControlAction, ScenarioState


class Scenario(ABC):
    """Abstract scenario with reset and step lifecycle."""

    @abstractmethod
    def reset(self) -> ScenarioState:
        """Reset the scenario and return the initial state."""

    @abstractmethod
    def step(self, action: ControlAction) -> ScenarioState:
        """Advance the scenario by one step."""

    @abstractmethod
    def get_state(self) -> ScenarioState:
        """Return the current scenario state."""
