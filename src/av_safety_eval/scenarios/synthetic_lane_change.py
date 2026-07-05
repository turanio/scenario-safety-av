"""Synthetic lane-change-like scenario for tests and demos."""

from __future__ import annotations

from av_safety_eval.scenarios.synthetic_interaction import (
    SyntheticInteractionScenario,
    SyntheticScenarioConfig,
)


class SyntheticLaneChangeScenario(SyntheticInteractionScenario):
    """A deterministic scenario with ego behind a slower merging target."""

    def __init__(self, dt: float = 0.2, max_steps: int = 50) -> None:
        super().__init__(
            SyntheticScenarioConfig(
                name="synthetic_lane_change",
                ego_x=0.0,
                ego_y=0.0,
                ego_vx=10.0,
                ego_vy=0.0,
                target_x=18.0,
                target_y=3.5,
                target_vx=7.0,
                target_vy=-0.35,
                dt=dt,
                horizon_steps=max_steps,
            )
        )
