import csv

import numpy as np
import pytest

from av_safety_eval.carla.carla_client import CarlaClientConfig, CarlaSession
from av_safety_eval.carla.metrics import CarlaPolicyMetrics, PolicyMetricTracker
from av_safety_eval.carla.scenarios import (
    HiddenRiskCarlaScenario,
    HiddenRiskScenarioConfig,
    ScenarioGeometry,
    build_hidden_risk_scenario_suite,
)
from av_safety_eval.carla.vehicle_controller import SimpleLongitudinalController
from av_safety_eval.experiments.run_carla_hidden_risk_validation import (
    SUITE_RESULT_FIELDS,
    _suite_summary_markdown,
    _write_suite_results,
)
from av_safety_eval.planning.safety_filter import (
    BRAKE,
    NO_BRAKE,
    evaluate_probability_aware_filter,
    evaluate_top1_filter,
    evaluate_worst_case_filter,
)


def test_metric_tracker_counts_braking_episodes_and_near_miss() -> None:
    tracker = PolicyMetricTracker(near_miss_threshold_m=3.0)
    samples = [
        (0.05, 6.0, 10.0, False),
        (0.10, 2.5, 9.0, True),
        (0.15, 2.0, 8.0, True),
        (0.20, 3.5, 8.0, False),
        (0.25, 4.0, 7.0, True),
    ]
    for time_seconds, distance, speed, braking in samples:
        tracker.update(
            time_seconds=time_seconds,
            center_distance_m=distance,
            ego_speed_mps=speed,
            braking=braking,
        )

    result = tracker.finalize(
        policy_name="test_policy",
        expected_steps=len(samples),
        collision_sensor_available=False,
        collision_note="sensor unavailable in unit test",
    )

    assert result.minimum_distance == pytest.approx(2.0)
    assert result.near_miss is True
    assert result.collision is False
    assert result.number_of_braking_interventions == 2
    assert result.first_braking_time == pytest.approx(0.10)
    assert result.final_ego_speed == pytest.approx(7.0)
    assert result.scenario_success is True
    assert "unavailable" in result.collision_note.lower()


def test_metric_tracker_uses_collision_event_when_sensor_available() -> None:
    tracker = PolicyMetricTracker()
    tracker.update(
        time_seconds=0.05,
        center_distance_m=1.0,
        ego_speed_mps=2.0,
        braking=True,
        collision_detected=True,
    )
    result = tracker.finalize(
        policy_name="test_policy",
        expected_steps=1,
        collision_sensor_available=True,
        collision_note="CARLA collision sensor",
    )

    assert result.collision is True
    assert result.scenario_success is False


def test_controller_brakes_or_maintains_speed_without_carla() -> None:
    controller = SimpleLongitudinalController(target_speed_mps=12.0)

    braking = controller.command(10.0, should_brake=True)
    maintaining = controller.command(10.0, should_brake=False)

    assert braking.throttle == 0.0
    assert braking.brake == pytest.approx(0.8)
    assert maintaining.brake == 0.0
    assert 0.0 < maintaining.throttle <= controller.max_throttle


def test_synthetic_modes_are_testable_without_carla() -> None:
    config = HiddenRiskScenarioConfig()
    scenario = HiddenRiskCarlaScenario(config)
    scenario.geometry = ScenarioGeometry(
        origin_x=0.0,
        origin_y=0.0,
        origin_z=0.0,
        yaw_degrees=0.0,
        forward_xy=np.array([1.0, 0.0]),
        right_xy=np.array([0.0, 1.0]),
        target_lateral_offset_m=3.5,
    )

    modes = scenario.synthetic_future_modes(time_seconds=0.0)

    assert modes.positions.shape == (3, config.prediction_steps, 2)
    assert modes.names == (
        "safe_continuation",
        "moderate_cut_in",
        "aggressive_cut_in",
    )
    assert modes.probabilities.sum() == pytest.approx(1.0)
    assert modes.positions[0, -1, 1] == pytest.approx(3.5)
    assert abs(modes.positions[2, -1, 1]) < 1e-9


def test_initial_hypotheses_create_hidden_risk_policy_disagreement() -> None:
    config = HiddenRiskScenarioConfig()
    scenario = HiddenRiskCarlaScenario(config)
    scenario.geometry = ScenarioGeometry(
        origin_x=0.0,
        origin_y=0.0,
        origin_z=0.0,
        yaw_degrees=0.0,
        forward_xy=np.array([1.0, 0.0]),
        right_xy=np.array([0.0, 1.0]),
        target_lateral_offset_m=3.5,
    )
    modes = scenario.synthetic_future_modes(time_seconds=0.0)
    times = (
        np.arange(1, config.prediction_steps + 1, dtype=float)
        * config.fixed_delta_seconds
    )
    ego_future = np.column_stack(
        [config.ego_target_speed_mps * times, np.zeros_like(times)]
    )
    distances = np.linalg.norm(modes.positions - ego_future[None, :, :], axis=-1)

    top1 = evaluate_top1_filter(distances, modes.probabilities)
    worst_case = evaluate_worst_case_filter(distances, modes.probabilities)
    probability_aware = evaluate_probability_aware_filter(
        distances,
        modes.probabilities,
        probability_threshold=config.probability_threshold,
    )

    assert top1.action == NO_BRAKE
    assert worst_case.action == BRAKE
    assert probability_aware.action == BRAKE
    assert probability_aware.trigger_mode == 1


def test_carla_session_tick_uses_keyword_timeout() -> None:
    class FakeWorld:
        def __init__(self) -> None:
            self.timeout = None

        def tick(self, *, timeout: float) -> int:
            self.timeout = timeout
            return 42

    session = CarlaSession(CarlaClientConfig(tick_timeout_seconds=7.5))
    session.world = FakeWorld()

    assert session.tick() == 42
    assert session.world.timeout == pytest.approx(7.5)


def test_carla_session_tick_falls_back_to_seconds_keyword() -> None:
    class FakeWorld:
        def __init__(self) -> None:
            self.seconds = None

        def tick(self, *, seconds: float) -> int:
            self.seconds = seconds
            return 43

    session = CarlaSession(CarlaClientConfig(tick_timeout_seconds=8.0))
    session.world = FakeWorld()

    assert session.tick() == 43
    assert session.world.seconds == pytest.approx(8.0)


def test_carla_session_tick_reports_timeout_failure() -> None:
    class FakeWorld:
        def tick(self, *, timeout: float) -> int:
            raise RuntimeError("server time-out")

    session = CarlaSession(CarlaClientConfig(tick_timeout_seconds=3.0))
    session.world = FakeWorld()

    with pytest.raises(RuntimeError, match="tick 1 failed or timed out after 3.0"):
        session.tick()


def test_carla_session_refuses_unbounded_tick_fallback() -> None:
    class FakeWorld:
        def tick(self) -> int:
            return 44

    session = CarlaSession(CarlaClientConfig())
    session.world = FakeWorld()

    with pytest.raises(RuntimeError, match="does not accept a bounded timeout"):
        session.tick()


def test_scenario_suite_has_expected_initial_policy_decisions() -> None:
    expected_actions = {
        "hidden_low_probability": (NO_BRAKE, BRAKE, NO_BRAKE),
        "borderline_probability_aware": (NO_BRAKE, BRAKE, BRAKE),
        "near_miss_style": (BRAKE, BRAKE, BRAKE),
    }
    variants = build_hidden_risk_scenario_suite()

    assert tuple(variant.name for variant in variants) == tuple(expected_actions)
    for variant in variants:
        config = variant.config
        scenario = HiddenRiskCarlaScenario(config)
        scenario.geometry = ScenarioGeometry(
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
            yaw_degrees=0.0,
            forward_xy=np.array([1.0, 0.0]),
            right_xy=np.array([0.0, 1.0]),
            target_lateral_offset_m=3.5,
        )
        modes = scenario.synthetic_future_modes(time_seconds=0.0)
        times = (
            np.arange(1, config.prediction_steps + 1, dtype=float)
            * config.fixed_delta_seconds
        )
        ego_future = np.column_stack(
            [config.ego_target_speed_mps * times, np.zeros_like(times)]
        )
        distances = np.linalg.norm(
            modes.positions - ego_future[None, :, :],
            axis=-1,
        )
        actions = (
            evaluate_top1_filter(distances, modes.probabilities).action,
            evaluate_worst_case_filter(distances, modes.probabilities).action,
            evaluate_probability_aware_filter(
                distances,
                modes.probabilities,
                probability_threshold=config.probability_threshold,
            ).action,
        )

        assert actions == expected_actions[variant.name]


def test_scenario_suite_writes_thesis_facing_outputs(tmp_path) -> None:
    variant = build_hidden_risk_scenario_suite()[0]
    metric = CarlaPolicyMetrics(
        policy_name="top1_policy",
        minimum_distance=2.5,
        near_miss=True,
        collision=False,
        collision_note="CARLA collision sensor",
        number_of_braking_interventions=1,
        first_braking_time=1.25,
        final_ego_speed=4.0,
        scenario_success=True,
    )
    csv_path = tmp_path / "suite.csv"

    _write_suite_results(csv_path, [(variant, metric)])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == list(SUITE_RESULT_FIELDS)
    assert rows[0]["scenario_name"] == "hidden_low_probability"
    assert rows[0]["actual_target_behavior"] == "aggressive_cut_in"
    assert rows[0]["mode_probabilities"] == "[0.93, 0.03, 0.04]"

    summary = _suite_summary_markdown(
        [(variant, metric)],
        (variant,),
        CarlaClientConfig(),
    )
    assert "not intended as a general CARLA benchmark" in summary
    assert "QCNet/AV2 500-scenario evaluation" in summary
    assert "online QCNet" in summary
