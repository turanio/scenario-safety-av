import csv
from pathlib import Path

import pytest

from av_safety_eval.experiments.run_uncertainty_planner_comparison import (
    UNCERTAINTY_SUMMARY_COLUMNS,
    run_uncertainty_planner_comparison,
)
from av_safety_eval.planners.conservative_uncertainty_planner import ConservativeUncertaintyPlanner
from av_safety_eval.planners.standard_planner import StandardPlanner
from av_safety_eval.predictors.constant_velocity import ConstantVelocityPredictor
from av_safety_eval.predictors.synthetic_multimodal import SyntheticMultimodalPredictor
from av_safety_eval.scenarios.delayed_cut_in import DelayedCutInScenario
from av_safety_eval.scenarios.synthetic_interaction import (
    SyntheticInteractionScenario,
    ambiguous_cut_in_config,
    delayed_cut_in_config,
)


def _by_scenario_and_planner(results: list[dict]) -> dict[tuple[str, str], dict]:
    return {(result["scenario"], result["planner"]): result for result in results}


def test_synthetic_multimodal_predictor_returns_multiple_trajectories() -> None:
    config = ambiguous_cut_in_config()
    state = SyntheticInteractionScenario(config).reset()
    predictor = SyntheticMultimodalPredictor()

    prediction = predictor.predict([state.agents[0]], horizon_steps=config.horizon_steps, dt=config.dt)

    assert len(prediction.trajectories) > 1
    assert prediction.probabilities is not None
    assert len(prediction.probabilities) == len(prediction.trajectories)
    assert prediction.trajectories[0].positions.shape == (config.horizon_steps, 2)


def test_uncertainty_planner_reacts_when_one_mode_is_risky() -> None:
    config = ambiguous_cut_in_config()
    state = SyntheticInteractionScenario(config).reset()
    prediction = SyntheticMultimodalPredictor().predict(
        [state.agents[0]],
        horizon_steps=config.horizon_steps,
        dt=config.dt,
    )

    action = ConservativeUncertaintyPlanner().plan(state, [prediction])

    assert action.acceleration < 0.0


def test_standard_planner_reacts_less_in_ambiguous_cut_in() -> None:
    config = ambiguous_cut_in_config()
    state = SyntheticInteractionScenario(config).reset()
    standard_prediction = ConstantVelocityPredictor().predict(
        [state.agents[0]],
        horizon_steps=config.horizon_steps,
        dt=config.dt,
    )
    multimodal_prediction = SyntheticMultimodalPredictor().predict(
        [state.agents[0]],
        horizon_steps=config.horizon_steps,
        dt=config.dt,
    )

    standard_action = StandardPlanner().plan(state, [standard_prediction])
    uncertainty_action = ConservativeUncertaintyPlanner().plan(state, [multimodal_prediction])

    assert standard_action.acceleration == 0.0
    assert uncertainty_action.acceleration < standard_action.acceleration


def test_uncertainty_runner_creates_summary_csv(tmp_path: Path) -> None:
    results = run_uncertainty_planner_comparison(output_root=tmp_path)

    summary_file = tmp_path / "metrics" / "uncertainty_planner_comparison_summary.csv"
    assert summary_file.exists()
    with summary_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == UNCERTAINTY_SUMMARY_COLUMNS
    assert len(rows) == 4
    assert all(Path(result["metrics_file"]).exists() for result in results)
    assert all(Path(result["log_file"]).exists() for result in results)


def test_uncertainty_runner_shows_behavioral_difference(tmp_path: Path) -> None:
    results = run_uncertainty_planner_comparison(output_root=tmp_path)
    by_case = _by_scenario_and_planner(results)

    standard = by_case[("ambiguous_cut_in", "standard")]
    uncertainty = by_case[("ambiguous_cut_in", "uncertainty_aware_conservative")]
    assert standard["prediction_modes"] == 1
    assert uncertainty["prediction_modes"] > 1
    assert standard["intervention_count"] == 0
    assert uncertainty["intervention_count"] > standard["intervention_count"]
    assert uncertainty["min_distance"] > standard["min_distance"]


def test_delayed_cut_in_changes_target_y_after_delay() -> None:
    config = delayed_cut_in_config()
    scenario = DelayedCutInScenario(config, cut_in_delay=1.0)
    state = scenario.reset()
    initial_y = state.agents[0].y

    for _ in range(5):
        state = scenario.step(action=StandardPlanner().plan(state, predictions=[]))
    assert state.time_seconds == pytest.approx(1.0)
    assert state.agents[0].y == initial_y

    state = scenario.step(action=StandardPlanner().plan(state, predictions=[]))
    assert state.time_seconds == pytest.approx(1.2)
    assert state.agents[0].y < initial_y


def test_uncertainty_runner_includes_delayed_cut_in(tmp_path: Path) -> None:
    results = run_uncertainty_planner_comparison(output_root=tmp_path)

    assert "delayed_cut_in" in {result["scenario"] for result in results}


def test_uncertainty_planner_improves_delayed_cut_in(tmp_path: Path) -> None:
    results = run_uncertainty_planner_comparison(output_root=tmp_path)
    by_case = _by_scenario_and_planner(results)

    standard = by_case[("delayed_cut_in", "standard")]
    uncertainty = by_case[("delayed_cut_in", "uncertainty_aware_conservative")]
    assert uncertainty["intervention_count"] > standard["intervention_count"]
    assert uncertainty["min_distance"] > standard["min_distance"]
    assert standard["near_miss"] is True
    assert uncertainty["near_miss"] is False
