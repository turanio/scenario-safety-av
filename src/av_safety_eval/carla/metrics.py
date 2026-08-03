"""CARLA run metrics that remain testable without the CARLA package."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CarlaPolicyMetrics:
    policy_name: str
    minimum_distance: float
    near_miss: bool
    collision: bool
    collision_note: str
    number_of_braking_interventions: int
    first_braking_time: float | None
    final_ego_speed: float
    scenario_success: bool

    def to_dict(self) -> dict:
        row = asdict(self)
        row["first_braking_time"] = (
            "" if self.first_braking_time is None else self.first_braking_time
        )
        return row


class PolicyMetricTracker:
    """Accumulate one policy rollout and count distinct braking episodes."""

    def __init__(self, near_miss_threshold_m: float = 3.0) -> None:
        if near_miss_threshold_m <= 0 or not np.isfinite(near_miss_threshold_m):
            raise ValueError("near_miss_threshold_m must be finite and positive")
        self.near_miss_threshold_m = float(near_miss_threshold_m)
        self.times: list[float] = []
        self.distances: list[float] = []
        self.ego_speeds: list[float] = []
        self._braking_interventions = 0
        self._braking_last_step = False
        self._first_braking_time: float | None = None
        self._collision_seen = False

    def update(
        self,
        *,
        time_seconds: float,
        center_distance_m: float,
        ego_speed_mps: float,
        braking: bool,
        collision_detected: bool = False,
    ) -> None:
        values = (time_seconds, center_distance_m, ego_speed_mps)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("metric samples must be finite")
        if time_seconds < 0 or center_distance_m < 0 or ego_speed_mps < 0:
            raise ValueError("time, distance, and speed must be non-negative")

        self.times.append(float(time_seconds))
        self.distances.append(float(center_distance_m))
        self.ego_speeds.append(float(ego_speed_mps))
        if braking and not self._braking_last_step:
            self._braking_interventions += 1
            if self._first_braking_time is None:
                self._first_braking_time = float(time_seconds)
        self._braking_last_step = bool(braking)
        self._collision_seen = self._collision_seen or bool(collision_detected)

    def finalize(
        self,
        *,
        policy_name: str,
        expected_steps: int,
        collision_sensor_available: bool,
        collision_note: str,
    ) -> CarlaPolicyMetrics:
        if not policy_name:
            raise ValueError("policy_name must not be empty")
        if expected_steps <= 0:
            raise ValueError("expected_steps must be positive")
        if not self.distances:
            raise ValueError("at least one metric sample is required")

        collision = self._collision_seen if collision_sensor_available else False
        note = collision_note
        if not collision_sensor_available and "unavailable" not in note.lower():
            note = "Collision sensor unavailable; collision is reported false. " + note
        completed = len(self.distances) == expected_steps
        minimum_distance = float(min(self.distances))
        return CarlaPolicyMetrics(
            policy_name=policy_name,
            minimum_distance=round(minimum_distance, 6),
            near_miss=minimum_distance < self.near_miss_threshold_m,
            collision=collision,
            collision_note=note,
            number_of_braking_interventions=self._braking_interventions,
            first_braking_time=(
                None
                if self._first_braking_time is None
                else round(self._first_braking_time, 6)
            ),
            final_ego_speed=round(float(self.ego_speeds[-1]), 6),
            scenario_success=completed and not collision,
        )
