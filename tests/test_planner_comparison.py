import csv
from pathlib import Path

from av_safety_eval.common.types import AgentState, ScenarioState
from av_safety_eval.experiments.run_planner_comparison import (
    PLANNER_COMPARISON_SUMMARY_COLUMNS,
    run_planner_comparison,
)
from av_safety_eval.planners.naive_planner import NaivePlanner


def _by_planner_and_scenario(results: list[dict]) -> dict[tuple[str, str], dict]:
    return {(result["planner"], result["scenario"]): result for result in results}


def test_naive_planner_always_maintains_speed() -> None:
    planner = NaivePlanner()
    state = ScenarioState(
        ego=AgentState(agent_id="ego", x=0.0, y=0.0, vx=1.0, vy=0.0),
        agents=[],
        dt=0.2,
    )

    action = planner.plan(state, predictions=[])

    assert action.acceleration == 0.0
    assert action.steering == 0.0


def test_planner_comparison_runner_works_and_creates_summary(tmp_path: Path) -> None:
    results = run_planner_comparison(output_root=tmp_path)

    assert len(results) == 8
    assert {result["planner"] for result in results} == {"naive", "standard"}
    summary_file = tmp_path / "metrics" / "planner_comparison_summary.csv"
    assert summary_file.exists()
    with summary_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == PLANNER_COMPARISON_SUMMARY_COLUMNS
    assert len(rows) == 8


def test_planner_comparison_outputs_per_run_artifacts(tmp_path: Path) -> None:
    results = run_planner_comparison(output_root=tmp_path)

    for result in results:
        assert Path(result["metrics_file"]).exists()
        assert Path(result["log_file"]).exists()


def test_naive_has_zero_interventions(tmp_path: Path) -> None:
    results = run_planner_comparison(output_root=tmp_path)

    assert all(result["intervention_count"] == 0 for result in results if result["planner"] == "naive")


def test_standard_intervenes_and_improves_collision_risk_distance(tmp_path: Path) -> None:
    results = run_planner_comparison(output_root=tmp_path)
    by_case = _by_planner_and_scenario(results)

    naive = by_case[("naive", "collision_risk_cut_in")]
    standard = by_case[("standard", "collision_risk_cut_in")]
    assert standard["intervention_count"] >= 1
    assert standard["min_distance"] > naive["min_distance"]
    assert naive["success"] is False
    assert standard["success"] is True


def test_no_interaction_remains_safe_for_both_planners(tmp_path: Path) -> None:
    results = run_planner_comparison(output_root=tmp_path)
    by_case = _by_planner_and_scenario(results)

    assert by_case[("naive", "no_interaction")]["success"] is True
    assert by_case[("standard", "no_interaction")]["success"] is True
    assert by_case[("naive", "no_interaction")]["collision"] is False
    assert by_case[("standard", "no_interaction")]["collision"] is False
