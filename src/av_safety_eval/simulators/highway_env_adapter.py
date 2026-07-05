"""Optional highway-env adapter placeholder."""

from __future__ import annotations

from typing import Any

from av_safety_eval.common.types import ControlAction, ScenarioState
from av_safety_eval.simulators.base import SimulatorAdapter


class HighwayEnvAdapter(SimulatorAdapter):
    """Adapter boundary for highway-env.

    The optional dependency is imported only when this adapter is constructed.
    Observation conversion will be implemented in a later simulator task.
    """

    def __init__(self, env_id: str = "highway-v0", config: dict[str, Any] | None = None) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "highway-env support requires optional dependencies: "
                "pip install -e '.[sim]'"
            ) from exc

        self.env = gym.make(env_id)
        self.config = config or {}

    def reset(self) -> ScenarioState:
        raise NotImplementedError("highway-env observation conversion is not implemented yet.")

    def step(self, action: ControlAction) -> ScenarioState:
        raise NotImplementedError("highway-env action and observation conversion is not implemented yet.")
