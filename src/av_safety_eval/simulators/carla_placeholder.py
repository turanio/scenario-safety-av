"""Placeholder for future CARLA simulator integration."""

from __future__ import annotations

from av_safety_eval.common.types import ControlAction, ScenarioState
from av_safety_eval.simulators.base import SimulatorAdapter


class CarlaAdapterPlaceholder(SimulatorAdapter):
    """Explicitly marks CARLA as a later validation target."""

    def reset(self) -> ScenarioState:
        raise NotImplementedError("CARLA integration is reserved for final validation work.")

    def step(self, action: ControlAction) -> ScenarioState:
        raise NotImplementedError("CARLA integration is reserved for final validation work.")
