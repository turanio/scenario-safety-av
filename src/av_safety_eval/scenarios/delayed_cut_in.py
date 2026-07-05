"""Delayed cut-in synthetic scenario."""

from __future__ import annotations

from av_safety_eval.common.types import AgentState, ControlAction, ScenarioState
from av_safety_eval.scenarios.base import Scenario
from av_safety_eval.scenarios.synthetic_interaction import SyntheticScenarioConfig


class DelayedCutInScenario(Scenario):
    """Target holds an adjacent lane, then cuts into the ego lane after a delay."""

    def __init__(
        self,
        config: SyntheticScenarioConfig,
        cut_in_delay: float = 1.0,
        target_lane_y: float = 0.0,
        target_cut_in_vy: float = -1.2,
    ) -> None:
        if cut_in_delay < 0.0:
            raise ValueError("cut_in_delay must be non-negative.")
        if target_cut_in_vy >= 0.0:
            raise ValueError("target_cut_in_vy must move toward the ego lane.")
        self.config = config
        self.cut_in_delay = cut_in_delay
        self.target_lane_y = target_lane_y
        self.target_cut_in_vy = target_cut_in_vy
        self._state: ScenarioState | None = None

    def reset(self) -> ScenarioState:
        self._state = ScenarioState(
            ego=self.config.ego_state(),
            agents=[self.config.target_state()],
            time_step=0,
            dt=self.config.dt,
            metadata={
                "scenario": self.config.name,
                "cut_in_delay": self.cut_in_delay,
                "done": False,
            },
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

        target_vy = self._target_vy_for_current_time(target)
        next_target_y = max(self.target_lane_y, target.y + target_vy * dt)
        if next_target_y <= self.target_lane_y:
            target_vy = 0.0
        next_target = AgentState(
            agent_id=target.agent_id,
            x=target.x + target.vx * dt,
            y=next_target_y,
            vx=target.vx,
            vy=target_vy,
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
            metadata={
                "scenario": self.config.name,
                "cut_in_delay": self.cut_in_delay,
                "done": done,
            },
        )
        return self._state

    def get_state(self) -> ScenarioState:
        if self._state is None:
            raise RuntimeError("Scenario must be reset before reading state.")
        return self._state

    def _target_vy_for_current_time(self, target: AgentState) -> float:
        if self._state is None:
            raise RuntimeError("Scenario must be reset before reading target velocity.")
        if self._state.time_seconds < self.cut_in_delay:
            return 0.0
        if target.y <= self.target_lane_y:
            return 0.0
        return self.target_cut_in_vy
