"""Naive baseline planner."""

from __future__ import annotations

from av_safety_eval.common.types import ControlAction, PredictionSet, ScenarioState
from av_safety_eval.planners.base import Planner


class NaivePlanner(Planner):
    """Planner that always maintains current speed and lane."""

    def plan(self, state: ScenarioState, predictions: list[PredictionSet]) -> ControlAction:
        return ControlAction(acceleration=0.0, steering=0.0)
