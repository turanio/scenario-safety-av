from pathlib import Path

import pytest

from av_safety_eval.experiments.prepare_qcnet_validation_cohort import (
    enumerate_scenario_ids,
    select_independent_cohort,
)


def test_independent_cohort_is_seeded_sorted_unique_and_disjoint() -> None:
    available = [f"scenario-{index:03d}" for index in range(30)]
    historical = available[:5]

    first, summary = select_independent_cohort(available, historical, 10, seed=42)
    second, _ = select_independent_cohort(available, historical, 10, seed=42)

    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 10
    assert set(first).isdisjoint(historical)
    assert summary["remaining_candidate_ids"] == 25
    assert summary["historical_overlap_count"] == 0


def test_independent_cohort_rejects_insufficient_candidates() -> None:
    with pytest.raises(ValueError, match="cannot select 3"):
        select_independent_cohort(["a", "b"], ["a"], 3, seed=42)


def test_enumerate_scenario_ids_uses_directories_only(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "ignored.txt").write_text("not a scenario", encoding="utf-8")

    assert enumerate_scenario_ids(tmp_path) == ["a", "b"]
