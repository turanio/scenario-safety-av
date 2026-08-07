"""Controlled CARLA interaction inspired by QCNet hidden-risk patterns."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from av_safety_eval.carla.carla_client import CarlaSession


ACTUAL_TARGET_BEHAVIORS = (
    "safe_continuation",
    "moderate_cut_in",
    "aggressive_cut_in",
)


@dataclass(frozen=True)
class HiddenRiskScenarioConfig:
    """Parameters for one deterministic adjacent-lane cut-in interaction."""

    duration_seconds: float = 6.0
    fixed_delta_seconds: float = 0.05
    prediction_horizon_seconds: float = 4.0
    ego_target_speed_mps: float = 12.0
    target_speed_mps: float = 8.0
    initial_longitudinal_gap_m: float = 12.0
    cut_in_start_seconds: float = 1.0
    cut_in_duration_seconds: float = 1.5
    near_miss_threshold_m: float = 3.0
    probability_threshold: float = 0.05
    mode_probabilities: tuple[float, float, float] = (0.78, 0.18, 0.04)
    actual_target_behavior: str = "moderate_cut_in"

    def __post_init__(self) -> None:
        positive = {
            "duration_seconds": self.duration_seconds,
            "fixed_delta_seconds": self.fixed_delta_seconds,
            "prediction_horizon_seconds": self.prediction_horizon_seconds,
            "ego_target_speed_mps": self.ego_target_speed_mps,
            "target_speed_mps": self.target_speed_mps,
            "initial_longitudinal_gap_m": self.initial_longitudinal_gap_m,
            "cut_in_duration_seconds": self.cut_in_duration_seconds,
            "near_miss_threshold_m": self.near_miss_threshold_m,
        }
        for name, value in positive.items():
            if value <= 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")
        if self.cut_in_start_seconds < 0:
            raise ValueError("cut_in_start_seconds must be non-negative")
        if not 0 <= self.probability_threshold <= 1:
            raise ValueError("probability_threshold must be in [0, 1]")
        probabilities = np.asarray(self.mode_probabilities, dtype=float)
        if probabilities.shape != (3,) or np.any(probabilities < 0):
            raise ValueError("mode_probabilities must contain three non-negative values")
        if not np.isclose(float(probabilities.sum()), 1.0):
            raise ValueError("mode_probabilities must sum to one")
        if self.actual_target_behavior not in ACTUAL_TARGET_BEHAVIORS:
            allowed = ", ".join(ACTUAL_TARGET_BEHAVIORS)
            raise ValueError(f"actual_target_behavior must be one of: {allowed}")

    @property
    def num_steps(self) -> int:
        return int(round(self.duration_seconds / self.fixed_delta_seconds))

    @property
    def prediction_steps(self) -> int:
        return int(round(self.prediction_horizon_seconds / self.fixed_delta_seconds))


@dataclass(frozen=True)
class HiddenRiskScenarioVariant:
    """Named controlled scenario and its thesis-safe interpretation."""

    name: str
    inspired_by: str
    config: HiddenRiskScenarioConfig
    interpretation: str


def build_hidden_risk_scenario_suite(
    duration_seconds: float = 6.0,
) -> tuple[HiddenRiskScenarioVariant, ...]:
    """Return the three small controlled variants used in CARLA validation."""

    return (
        HiddenRiskScenarioVariant(
            name="hidden_low_probability",
            inspired_by="AV2 scenario 001749",
            config=HiddenRiskScenarioConfig(
                duration_seconds=duration_seconds,
                cut_in_duration_seconds=0.8,
                mode_probabilities=(0.93, 0.03, 0.04),
                actual_target_behavior="aggressive_cut_in",
            ),
            interpretation=(
                "The aggressive cut-in hypothesis is below the probability cutoff, "
                "so the probability-aware policy is expected to behave closer to "
                "top-1 than to the worst-case policy."
            ),
        ),
        HiddenRiskScenarioVariant(
            name="borderline_probability_aware",
            inspired_by="AV2 scenario 00e2cd",
            config=HiddenRiskScenarioConfig(
                duration_seconds=duration_seconds,
                cut_in_duration_seconds=1.5,
                mode_probabilities=(0.82, 0.14, 0.04),
                actual_target_behavior="moderate_cut_in",
            ),
            interpretation=(
                "The moderate cut-in hypothesis has probability 0.14 and passes "
                "the cutoff, so probability-aware and worst-case policies can "
                "respond before a top-1-only policy."
            ),
        ),
        HiddenRiskScenarioVariant(
            name="near_miss_style",
            inspired_by="AV2 scenario 003515",
            config=HiddenRiskScenarioConfig(
                duration_seconds=duration_seconds,
                cut_in_duration_seconds=1.5,
                mode_probabilities=(0.15, 0.70, 0.15),
                actual_target_behavior="moderate_cut_in",
            ),
            interpretation=(
                "The moderate cut-in is the highest-probability hypothesis, so all "
                "three policies are expected to brake; timing and resulting vehicle "
                "behavior remain the comparison of interest."
            ),
        ),
    )


@dataclass(frozen=True)
class ScenarioGeometry:
    origin_x: float
    origin_y: float
    origin_z: float
    yaw_degrees: float
    forward_xy: np.ndarray
    right_xy: np.ndarray
    target_lateral_offset_m: float


@dataclass(frozen=True)
class SyntheticFutureModes:
    """Three target futures in CARLA world coordinates."""

    names: tuple[str, str, str]
    positions: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if positions.ndim != 3 or positions.shape[0] != 3 or positions.shape[2] != 2:
            raise ValueError("positions must have shape (3, steps, 2)")
        if probabilities.shape != (3,):
            raise ValueError("probabilities must have shape (3,)")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True)
class ScenarioActors:
    ego: Any
    target: Any
    collision_recorder: Any


def _angle_difference_degrees(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _smoothstep(progress: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(progress, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


class HiddenRiskCarlaScenario:
    """Script a target cut-in while CARLA simulates the controlled ego vehicle."""

    MODE_NAMES = ("safe_continuation", "moderate_cut_in", "aggressive_cut_in")

    def __init__(self, config: HiddenRiskScenarioConfig) -> None:
        self.config = config
        self.geometry: ScenarioGeometry | None = None

    def _straight_geometry(self, session: CarlaSession) -> ScenarioGeometry:
        if session.world is None or session.carla is None:
            raise RuntimeError("CARLA session is not connected")
        road_map = session.world.get_map()
        carla = session.carla
        required_length = self.config.ego_target_speed_mps * self.config.duration_seconds + 10.0

        for transform in road_map.get_spawn_points():
            waypoint = road_map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                continue

            cursor = waypoint
            straight = True
            for _ in range(int(math.ceil(required_length / 10.0))):
                next_waypoints = cursor.next(10.0)
                if not next_waypoints:
                    straight = False
                    break
                cursor = min(
                    next_waypoints,
                    key=lambda item: _angle_difference_degrees(
                        item.transform.rotation.yaw,
                        waypoint.transform.rotation.yaw,
                    ),
                )
                if _angle_difference_degrees(
                    cursor.transform.rotation.yaw,
                    waypoint.transform.rotation.yaw,
                ) > 5.0:
                    straight = False
                    break
            if not straight:
                continue

            yaw_radians = math.radians(waypoint.transform.rotation.yaw)
            forward = np.array([math.cos(yaw_radians), math.sin(yaw_radians)], dtype=float)
            right = np.array([-forward[1], forward[0]], dtype=float)

            ahead_options = waypoint.next(self.config.initial_longitudinal_gap_m)
            if not ahead_options:
                continue
            ahead = min(
                ahead_options,
                key=lambda item: _angle_difference_degrees(
                    item.transform.rotation.yaw,
                    waypoint.transform.rotation.yaw,
                ),
            )
            for adjacent in (ahead.get_left_lane(), ahead.get_right_lane()):
                if adjacent is None or adjacent.lane_type != carla.LaneType.Driving:
                    continue
                if _angle_difference_degrees(
                    adjacent.transform.rotation.yaw,
                    waypoint.transform.rotation.yaw,
                ) > 30.0:
                    continue
                delta = np.array(
                    [
                        adjacent.transform.location.x - waypoint.transform.location.x,
                        adjacent.transform.location.y - waypoint.transform.location.y,
                    ]
                )
                lateral_offset = float(np.dot(delta, right))
                if abs(lateral_offset) < 2.0:
                    continue
                return ScenarioGeometry(
                    origin_x=float(waypoint.transform.location.x),
                    origin_y=float(waypoint.transform.location.y),
                    origin_z=float(waypoint.transform.location.z + 0.35),
                    yaw_degrees=float(waypoint.transform.rotation.yaw),
                    forward_xy=forward,
                    right_xy=right,
                    target_lateral_offset_m=lateral_offset,
                )

        raise RuntimeError(
            "Could not find a sufficiently straight CARLA road with an adjacent "
            "same-direction driving lane. Try Town04 or Town05."
        )

    def local_to_world(self, longitudinal_m: Any, lateral_m: Any) -> np.ndarray:
        if self.geometry is None:
            raise RuntimeError("Scenario geometry has not been initialized")
        longitudinal = np.asarray(longitudinal_m, dtype=float)
        lateral = np.asarray(lateral_m, dtype=float)
        origin = np.array([self.geometry.origin_x, self.geometry.origin_y])
        return (
            origin
            + longitudinal[..., None] * self.geometry.forward_xy
            + lateral[..., None] * self.geometry.right_xy
        )

    def spawn(self, session: CarlaSession) -> ScenarioActors:
        if session.carla is None:
            raise RuntimeError("CARLA session is not connected")
        self.geometry = self._straight_geometry(session)
        carla = session.carla

        ego_xy = self.local_to_world(0.0, 0.0)
        target_xy = self.local_to_world(
            self.config.initial_longitudinal_gap_m,
            self.geometry.target_lateral_offset_m,
        )
        rotation = carla.Rotation(yaw=self.geometry.yaw_degrees)
        ego_transform = carla.Transform(
            carla.Location(x=float(ego_xy[0]), y=float(ego_xy[1]), z=self.geometry.origin_z),
            rotation,
        )
        target_transform = carla.Transform(
            carla.Location(
                x=float(target_xy[0]),
                y=float(target_xy[1]),
                z=self.geometry.origin_z,
            ),
            rotation,
        )

        ego = session.spawn_vehicle(ego_transform, role_name="ego")
        target = session.spawn_vehicle(target_transform, role_name="scripted_target")
        target.set_simulate_physics(False)
        session.tick()

        ego.set_target_velocity(
            carla.Vector3D(
                x=float(self.geometry.forward_xy[0] * self.config.ego_target_speed_mps),
                y=float(self.geometry.forward_xy[1] * self.config.ego_target_speed_mps),
                z=0.0,
            )
        )
        self.update_target(target, 0.0, session)
        collision_recorder = session.attach_collision_sensor(ego)
        return ScenarioActors(ego, target, collision_recorder)

    def target_local_state(self, time_seconds: float) -> tuple[float, float]:
        if self.geometry is None:
            raise RuntimeError("Scenario geometry has not been initialized")
        if self.config.actual_target_behavior == "safe_continuation":
            lateral = self.geometry.target_lateral_offset_m
        else:
            progress = (
                (time_seconds - self.config.cut_in_start_seconds)
                / self.config.cut_in_duration_seconds
            )
            lateral = self.geometry.target_lateral_offset_m * (
                1.0 - float(_smoothstep(progress))
            )
        longitudinal = (
            self.config.initial_longitudinal_gap_m
            + self.config.target_speed_mps * time_seconds
        )
        return longitudinal, lateral

    def update_target(self, target: Any, time_seconds: float, session: CarlaSession) -> None:
        if self.geometry is None or session.carla is None:
            raise RuntimeError("Scenario has not been initialized")
        longitudinal, lateral = self.target_local_state(time_seconds)
        xy = self.local_to_world(longitudinal, lateral)
        target.set_transform(
            session.carla.Transform(
                session.carla.Location(
                    x=float(xy[0]),
                    y=float(xy[1]),
                    z=self.geometry.origin_z,
                ),
                session.carla.Rotation(yaw=self.geometry.yaw_degrees),
            )
        )

    def synthetic_future_modes(self, time_seconds: float) -> SyntheticFutureModes:
        if self.geometry is None:
            raise RuntimeError("Scenario geometry has not been initialized")
        current_longitudinal, current_lateral = self.target_local_state(time_seconds)
        future_times = (
            np.arange(1, self.config.prediction_steps + 1, dtype=float)
            * self.config.fixed_delta_seconds
        )
        longitudinal = current_longitudinal + self.config.target_speed_mps * future_times

        safe_lateral = np.full_like(future_times, current_lateral)
        moderate_progress = _smoothstep(future_times / 1.8)
        aggressive_progress = _smoothstep(future_times / 0.8)
        moderate_lateral = current_lateral * (1.0 - moderate_progress)
        aggressive_lateral = current_lateral * (1.0 - aggressive_progress)
        positions = np.stack(
            [
                self.local_to_world(longitudinal, lateral)
                for lateral in (safe_lateral, moderate_lateral, aggressive_lateral)
            ]
        )
        return SyntheticFutureModes(
            names=self.MODE_NAMES,
            positions=positions,
            probabilities=np.asarray(self.config.mode_probabilities, dtype=float),
        )

    def ego_constant_velocity_future(self, ego: Any) -> np.ndarray:
        location = ego.get_location()
        velocity = ego.get_velocity()
        times = (
            np.arange(1, self.config.prediction_steps + 1, dtype=float)
            * self.config.fixed_delta_seconds
        )
        origin = np.array([location.x, location.y], dtype=float)
        velocity_xy = np.array([velocity.x, velocity.y], dtype=float)
        return origin + times[:, None] * velocity_xy

    @staticmethod
    def center_distance(first: Any, second: Any) -> float:
        first_location = first.get_location()
        second_location = second.get_location()
        return float(
            math.hypot(
                first_location.x - second_location.x,
                first_location.y - second_location.y,
            )
        )
