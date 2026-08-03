"""Deterministic longitudinal control for the CARLA thesis experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LongitudinalControlCommand:
    """Simulator-neutral throttle and brake command."""

    throttle: float
    brake: float
    steer: float = 0.0


@dataclass(frozen=True)
class SimpleLongitudinalController:
    """Maintain a target speed or apply one fixed braking command."""

    target_speed_mps: float = 12.0
    speed_gain: float = 0.25
    max_throttle: float = 0.65
    brake_strength: float = 0.8

    def __post_init__(self) -> None:
        if self.target_speed_mps < 0:
            raise ValueError("target_speed_mps must be non-negative")
        if self.speed_gain <= 0:
            raise ValueError("speed_gain must be positive")
        if not 0 <= self.max_throttle <= 1:
            raise ValueError("max_throttle must be in [0, 1]")
        if not 0 <= self.brake_strength <= 1:
            raise ValueError("brake_strength must be in [0, 1]")

    def command(
        self, current_speed_mps: float, *, should_brake: bool
    ) -> LongitudinalControlCommand:
        if current_speed_mps < 0 or not np.isfinite(current_speed_mps):
            raise ValueError("current_speed_mps must be finite and non-negative")
        if should_brake:
            return LongitudinalControlCommand(
                throttle=0.0,
                brake=self.brake_strength,
            )

        speed_error = self.target_speed_mps - current_speed_mps
        throttle = float(np.clip(self.speed_gain * speed_error, 0.0, self.max_throttle))
        return LongitudinalControlCommand(throttle=throttle, brake=0.0)
