import csv
from pathlib import Path

from av_safety_eval.experiments.run_closed_loop_baseline import (
    CLOSED_LOOP_SUMMARY_COLUMNS,
    run_closed_loop_baseline,
)


def test_closed_loop_runner_works_without_external_data(tmp_path: Path) -> None:
    results = run_closed_loop_baseline(output_root=tmp_path)

    assert {result["scenario"] for result in results} == {
        "safe_following",
        "near_miss_lane_change",
        "collision_risk_cut_in",
        "no_interaction",
    }
    assert all(result["predictor"] == "constant_velocity" for result in results)
    assert all(result["planner"] == "standard" for result in results)
    assert all(Path(result["metrics_file"]).exists() for result in results)


def test_closed_loop_logs_are_created(tmp_path: Path) -> None:
    results = run_closed_loop_baseline(output_root=tmp_path)

    for result in results:
        log_file = Path(result["log_file"])
        assert log_file.exists()
        with log_file.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == result["steps"]
        assert {"step", "ego_x", "target_x", "action_acceleration", "min_distance"} <= set(rows[0])


def test_closed_loop_summary_csv_is_created(tmp_path: Path) -> None:
    run_closed_loop_baseline(output_root=tmp_path)

    summary_file = tmp_path / "metrics" / "closed_loop_baseline_summary.csv"
    assert summary_file.exists()
    with summary_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == CLOSED_LOOP_SUMMARY_COLUMNS
    assert len(rows) == 4


def test_collision_risk_has_planner_intervention(tmp_path: Path) -> None:
    results = run_closed_loop_baseline(output_root=tmp_path)
    by_scenario = {result["scenario"]: result for result in results}

    assert by_scenario["collision_risk_cut_in"]["intervention_count"] >= 1


def test_no_interaction_has_zero_or_minimal_intervention(tmp_path: Path) -> None:
    results = run_closed_loop_baseline(output_root=tmp_path)
    by_scenario = {result["scenario"]: result for result in results}

    assert by_scenario["no_interaction"]["intervention_count"] <= 1
