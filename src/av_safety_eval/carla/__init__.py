"""Lightweight CARLA support for controlled thesis validation experiments."""

from av_safety_eval.carla.metrics import CarlaPolicyMetrics, PolicyMetricTracker
from av_safety_eval.carla.scenarios import HiddenRiskScenarioConfig, SyntheticFutureModes
from av_safety_eval.carla.vehicle_controller import (
    LongitudinalControlCommand,
    SimpleLongitudinalController,
)

__all__ = [
    "CarlaPolicyMetrics",
    "HiddenRiskScenarioConfig",
    "LongitudinalControlCommand",
    "PolicyMetricTracker",
    "SimpleLongitudinalController",
    "SyntheticFutureModes",
]
