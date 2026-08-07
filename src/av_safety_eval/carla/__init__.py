"""Lightweight CARLA support for controlled thesis validation experiments."""

from av_safety_eval.carla.image_capture import (
    CameraCaptureConfig,
    LatestRgbFrame,
    key_frame_filename,
)
from av_safety_eval.carla.metrics import CarlaPolicyMetrics, PolicyMetricTracker
from av_safety_eval.carla.scenarios import (
    HiddenRiskScenarioConfig,
    HiddenRiskScenarioVariant,
    SyntheticFutureModes,
    build_hidden_risk_scenario_suite,
)
from av_safety_eval.carla.vehicle_controller import (
    LongitudinalControlCommand,
    SimpleLongitudinalController,
)

__all__ = [
    "CameraCaptureConfig",
    "CarlaPolicyMetrics",
    "HiddenRiskScenarioConfig",
    "HiddenRiskScenarioVariant",
    "LongitudinalControlCommand",
    "LatestRgbFrame",
    "PolicyMetricTracker",
    "SimpleLongitudinalController",
    "SyntheticFutureModes",
    "build_hidden_risk_scenario_suite",
    "key_frame_filename",
]
