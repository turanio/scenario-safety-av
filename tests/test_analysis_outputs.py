from pathlib import Path

import pandas as pd
import pytest

from av_safety_eval.analysis.derived_metrics import compute_log_derived_metrics
from av_safety_eval.experiments.analyze_results import run_analysis


def test_compute_log_derived_metrics_from_fake_log() -> None:
    log = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "time": [0.0, 0.2, 0.4],
            "ego_x": [0.0, 2.0, 3.8],
            "ego_vx": [10.0, 9.0, 8.0],
            "action_acceleration": [0.0, -1.0, -2.0],
            "min_distance": [5.0, 4.0, 3.0],
        }
    )

    metrics = compute_log_derived_metrics(log)

    assert metrics["first_intervention_step"] == 1
    assert metrics["first_intervention_time"] == pytest.approx(0.2)
    assert metrics["max_braking"] == pytest.approx(-2.0)
    assert metrics["mean_acceleration"] == pytest.approx(-1.0)
    assert metrics["average_ego_speed"] == pytest.approx(9.0)
    assert metrics["final_ego_x"] == pytest.approx(3.8)
    assert metrics["minimum_logged_distance"] == pytest.approx(3.0)


def test_first_intervention_time_is_none_without_braking() -> None:
    log = pd.DataFrame(
        {
            "step": [0, 1],
            "time": [0.0, 0.2],
            "ego_x": [0.0, 2.0],
            "ego_vx": [10.0, 10.0],
            "action_acceleration": [0.0, 0.0],
            "min_distance": [5.0, 4.5],
        }
    )

    metrics = compute_log_derived_metrics(log)

    assert metrics["first_intervention_step"] is None
    assert metrics["first_intervention_time"] is None


def test_analysis_runner_creates_outputs(tmp_path: Path) -> None:
    _write_minimal_analysis_inputs(tmp_path)

    outputs = run_analysis(tmp_path)

    assert (tmp_path / "results" / "figures").exists()
    assert (tmp_path / "results" / "tables").exists()
    assert (tmp_path / "results" / "analysis").exists()

    required_tables = [
        "planner_comparison_table.csv",
        "uncertainty_comparison_table.csv",
        "delayed_cut_in_derived_metrics.csv",
        "key_findings_table.csv",
    ]
    for table in required_tables:
        assert (tmp_path / "results" / "tables" / table).exists()

    required_figures = [
        "planner_comparison_min_distance.png",
        "planner_comparison_interventions.png",
        "uncertainty_comparison_min_distance.png",
        "delayed_cut_in_distance_over_time.png",
        "delayed_cut_in_action_over_time.png",
    ]
    for figure in required_figures:
        assert (tmp_path / "results" / "figures" / figure).exists()

    summary = tmp_path / "docs" / "results_summary.md"
    assert summary.exists()
    summary_text = summary.read_text(encoding="utf-8")
    assert "synthetic multimodal predictor" in summary_text
    assert "not a diffusion model" in summary_text
    assert "not a full SafeIO implementation" in summary_text
    assert "avoids the delayed cut-in near miss" in summary_text
    assert "first intervention at" in summary_text
    assert "results/tables/planner_comparison_table.csv" in summary_text
    assert outputs["manifest"].endswith("analysis_manifest.json")

    key_findings = pd.read_csv(tmp_path / "results" / "tables" / "key_findings_table.csv")
    assert list(key_findings.columns) == [
        "experiment_group",
        "scenario",
        "system",
        "predictor",
        "prediction_modes",
        "min_distance",
        "near_miss",
        "collision",
        "intervention_count",
        "first_intervention_time",
        "interpretation",
    ]
    assert "Uncertainty-aware" in set(key_findings["system"])
    assert "Delayed cut-in" in set(key_findings["scenario"])


def _write_minimal_analysis_inputs(root: Path) -> None:
    metrics_dir = root / "results" / "metrics"
    logs_dir = root / "results" / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "scenario": "collision_risk_cut_in",
                "planner": "naive",
                "predictor": "constant_velocity",
                "steps": 30,
                "final_time": 6.0,
                "min_distance": 0.2,
                "near_miss": True,
                "collision": True,
                "intervention_count": 0,
                "success": False,
            },
            {
                "scenario": "collision_risk_cut_in",
                "planner": "standard",
                "predictor": "constant_velocity",
                "steps": 30,
                "final_time": 6.0,
                "min_distance": 4.4,
                "near_miss": False,
                "collision": False,
                "intervention_count": 7,
                "success": True,
            },
        ]
    ).to_csv(metrics_dir / "planner_comparison_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "scenario": "delayed_cut_in",
                "planner": "standard",
                "predictor": "constant_velocity",
                "prediction_modes": 1,
                "steps": 30,
                "final_time": 6.0,
                "min_distance": 2.3,
                "near_miss": True,
                "collision": False,
                "intervention_count": 12,
                "success": True,
            },
            {
                "scenario": "delayed_cut_in",
                "planner": "uncertainty_aware_conservative",
                "predictor": "synthetic_multimodal",
                "prediction_modes": 3,
                "steps": 30,
                "final_time": 6.0,
                "min_distance": 3.8,
                "near_miss": False,
                "collision": False,
                "intervention_count": 15,
                "success": True,
            },
        ]
    ).to_csv(metrics_dir / "uncertainty_planner_comparison_summary.csv", index=False)

    standard_log = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "time": [0.0, 0.2, 0.4],
            "ego_x": [0.0, 2.0, 3.9],
            "ego_y": [0.0, 0.0, 0.0],
            "ego_vx": [10.0, 10.0, 9.0],
            "ego_vy": [0.0, 0.0, 0.0],
            "target_x": [5.0, 6.4, 7.8],
            "target_y": [3.5, 3.5, 3.2],
            "target_vx": [7.0, 7.0, 7.0],
            "target_vy": [0.0, 0.0, -1.2],
            "action_acceleration": [0.0, -1.0, -1.0],
            "action_steering": [0.0, 0.0, 0.0],
            "min_distance": [6.0, 4.0, 2.5],
            "time_to_collision": ["", "", ""],
            "near_miss": [False, False, True],
            "collision": [False, False, False],
        }
    )
    uncertainty_log = standard_log.copy()
    uncertainty_log["action_acceleration"] = [-2.0, -2.0, -2.0]
    uncertainty_log["ego_vx"] = [10.0, 9.6, 9.2]
    uncertainty_log["ego_x"] = [0.0, 1.96, 3.84]
    uncertainty_log["min_distance"] = [6.0, 4.5, 3.8]

    standard_log.to_csv(
        logs_dir / "uncertainty_comparison_standard_delayed_cut_in.csv",
        index=False,
    )
    uncertainty_log.to_csv(
        logs_dir / "uncertainty_comparison_uncertainty_aware_conservative_delayed_cut_in.csv",
        index=False,
    )
