from av_safety_eval.experiments.plot_qcnet_scenario_context_validation import (
    format_probability,
)
from av_safety_eval.experiments.plot_qcnet_server_hidden_risk_validation import (
    is_hidden_risk,
    select_hidden_risk_cases,
)


def _candidate(
    scenario_id: str,
    worst_distance: float,
    worst_probability: float,
    *,
    horizon_end: bool = False,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "ranking_row": {"worst_case_min_distance": worst_distance},
        "worst_case_mode_probability": worst_probability,
        "min_occurs_at_horizon_end": horizon_end,
        "map_context_available": True,
    }


def test_hidden_risk_uses_strict_brake_threshold() -> None:
    assert is_hidden_risk(
        {"top1_min_distance": 3.0, "worst_case_min_distance": 2.99}, 3.0
    )
    assert not is_hidden_risk(
        {"top1_min_distance": 2.99, "worst_case_min_distance": 2.0}, 3.0
    )
    assert not is_hidden_risk(
        {"top1_min_distance": 4.0, "worst_case_min_distance": 3.0}, 3.0
    )


def test_selection_balances_severity_probability_and_horizon() -> None:
    selected = select_hidden_risk_cases(
        [
            _candidate("tail", 0.2, 0.00001),
            _candidate("balanced", 0.5, 0.02),
            _candidate("probability", 2.8, 0.15),
            _candidate("final", 0.1, 0.4, horizon_end=True),
        ],
        meaningful_probability=0.01,
    )

    assert [row["scenario_id"] for row in selected] == [
        "balanced",
        "probability",
        "tail",
    ]
    assert [row["recommended_use"] for row in selected] == [
        "primary_case_study",
        "secondary_case",
        "appendix_only",
    ]


def test_small_probabilities_use_scientific_notation() -> None:
    assert format_probability(0.000007) == "7.00e-06"
    assert format_probability(0.024045) == "0.024"
