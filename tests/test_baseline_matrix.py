import csv
from pathlib import Path

from av_safety_eval.experiments.baseline_common import (
    BASELINE_SUMMARY_COLUMNS,
    evaluate_constant_velocity_baseline,
)
from av_safety_eval.experiments.run_baseline_matrix import run_baseline_matrix
from av_safety_eval.scenarios.synthetic_interaction import baseline_matrix_configs


def test_each_baseline_matrix_scenario_can_run(tmp_path: Path) -> None:
    results = [
        evaluate_constant_velocity_baseline(config, output_root=tmp_path)
        for config in baseline_matrix_configs()
    ]

    scenarios = {result["scenario"] for result in results}
    assert scenarios == {
        "safe_following",
        "near_miss_lane_change",
        "collision_risk_cut_in",
        "no_interaction",
    }
    for result in results:
        assert Path(result["metrics_file"]).exists()
        assert result["predictor"] == "constant_velocity"


def test_baseline_matrix_creates_aggregated_csv(tmp_path: Path) -> None:
    run_baseline_matrix(output_root=tmp_path)

    summary_file = tmp_path / "metrics" / "baseline_matrix_summary.csv"
    assert summary_file.exists()

    with summary_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert len(rows) == 4
    assert reader.fieldnames == BASELINE_SUMMARY_COLUMNS


def test_collision_risk_min_distance_is_smaller_than_safe_following(tmp_path: Path) -> None:
    results = run_baseline_matrix(output_root=tmp_path)
    by_scenario = {result["scenario"]: result for result in results}

    assert (
        by_scenario["collision_risk_cut_in"]["min_distance"]
        < by_scenario["safe_following"]["min_distance"]
    )
