"""Planner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from av_safety_eval.common.types import ControlAction, PredictionSet, ScenarioState


class Planner(ABC):
    """Abstract planner that maps scenario state and predictions to an action."""

    @abstractmethod
    def plan(self, state: ScenarioState, predictions: list[PredictionSet]) -> ControlAction:
        """Choose a control action."""
