"""Configurable synthetic two-agent scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from av_safety_eval.common.types import AgentState, ControlAction, ScenarioState
from av_safety_eval.scenarios.base import Scenario


@dataclass(frozen=True)
class SyntheticScenarioConfig:
    """Initial conditions for a deterministic ego-target interaction."""

    name: str
    ego_x: float
    ego_y: float
    ego_vx: float
    ego_vy: float
    target_x: float
    target_y: float
    target_vx: float
    target_vy: float
    dt: float = 0.2
    horizon_steps: int = 30

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")

    def ego_state(self) -> AgentState:
        """Create the initial ego state."""

        return AgentState(
            agent_id="ego",
            x=self.ego_x,
            y=self.ego_y,
            vx=self.ego_vx,
            vy=self.ego_vy,
        )

    def target_state(self) -> AgentState:
        """Create the initial target state."""

        return AgentState(
            agent_id="target",
            x=self.target_x,
            y=self.target_y,
            vx=self.target_vx,
            vy=self.target_vy,
        )


class SyntheticInteractionScenario(Scenario):
    """Deterministic two-vehicle scenario driven by initial-state config."""

    def __init__(self, config: SyntheticScenarioConfig) -> None:
        self.config = config
        self._state: ScenarioState | None = None

    def reset(self) -> ScenarioState:
        self._state = ScenarioState(
            ego=self.config.ego_state(),
            agents=[self.config.target_state()],
            time_step=0,
            dt=self.config.dt,
            metadata={"scenario": self.config.name, "done": False},
        )
        return self._state

    def step(self, action: ControlAction) -> ScenarioState:
        if self._state is None:
            raise RuntimeError("Scenario must be reset before stepping.")

        ego = self._state.ego
        target = self._state.agents[0]
        dt = self.config.dt

        next_ego_vx = max(0.0, ego.vx + action.acceleration * dt)
        next_ego = AgentState(
            agent_id=ego.agent_id,
            x=ego.x + ego.vx * dt + 0.5 * action.acceleration * dt**2,
            y=ego.y + (ego.vy + action.steering) * dt,
            vx=next_ego_vx,
            vy=ego.vy + action.steering,
            ax=action.acceleration,
            ay=0.0,
            heading=ego.heading,
        )
        next_target = AgentState(
            agent_id=target.agent_id,
            x=target.x + target.vx * dt,
            y=target.y + target.vy * dt,
            vx=target.vx,
            vy=target.vy,
            ax=0.0,
            ay=0.0,
            heading=target.heading,
        )

        next_time_step = self._state.time_step + 1
        done = next_time_step >= self.config.horizon_steps
        self._state = ScenarioState(
            ego=next_ego,
            agents=[next_target],
            time_step=next_time_step,
            dt=dt,
            metadata={"scenario": self.config.name, "done": done},
        )
        return self._state

    def get_state(self) -> ScenarioState:
        if self._state is None:
            raise RuntimeError("Scenario must be reset before reading state.")
        return self._state


def baseline_matrix_configs() -> list[SyntheticScenarioConfig]:
    """Return deterministic synthetic configs for the baseline matrix."""

    return [
        SyntheticScenarioConfig(
            name="safe_following",
            ego_x=0.0,
            ego_y=0.0,
            ego_vx=8.0,
            ego_vy=0.0,
            target_x=30.0,
            target_y=0.0,
            target_vx=8.5,
            target_vy=0.0,
            dt=0.2,
            horizon_steps=30,
        ),
        SyntheticScenarioConfig(
            name="near_miss_lane_change",
            ego_x=0.0,
            ego_y=0.0,
            ego_vx=10.0,
            ego_vy=0.0,
            target_x=16.0,
            target_y=3.5,
            target_vx=7.5,
            target_vy=-0.7,
            dt=0.2,
            horizon_steps=30,
        ),
        SyntheticScenarioConfig(
            name="collision_risk_cut_in",
            ego_x=0.0,
            ego_y=0.0,
            ego_vx=10.0,
            ego_vy=0.0,
            target_x=10.0,
            target_y=2.0,
            target_vx=7.0,
            target_vy=-0.6,
            dt=0.2,
            horizon_steps=30,
        ),
        SyntheticScenarioConfig(
            name="no_interaction",
            ego_x=0.0,
            ego_y=0.0,
            ego_vx=8.0,
            ego_vy=0.0,
            target_x=0.0,
            target_y=15.0,
            target_vx=8.0,
            target_vy=0.0,
            dt=0.2,
            horizon_steps=30,
        ),
    ]


def ambiguous_cut_in_config() -> SyntheticScenarioConfig:
    """Return a scenario where the actual path is safe but an alternative is risky."""

    return SyntheticScenarioConfig(
        name="ambiguous_cut_in",
        ego_x=0.0,
        ego_y=0.0,
        ego_vx=10.0,
        ego_vy=0.0,
        target_x=14.0,
        target_y=3.5,
        target_vx=7.5,
        target_vy=0.0,
        dt=0.2,
        horizon_steps=30,
    )


def delayed_cut_in_config() -> SyntheticScenarioConfig:
    """Return a scenario where the target cuts in after a short delay."""

    return SyntheticScenarioConfig(
        name="delayed_cut_in",
        ego_x=0.0,
        ego_y=0.0,
        ego_vx=10.0,
        ego_vy=0.0,
        target_x=5.0,
        target_y=3.5,
        target_vx=7.0,
        target_vy=0.0,
        dt=0.2,
        horizon_steps=30,
    )
