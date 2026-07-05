from av_safety_eval.common.types import ControlAction
from av_safety_eval.scenarios.synthetic_lane_change import SyntheticLaneChangeScenario


def test_synthetic_scenario_can_reset_and_step() -> None:
    scenario = SyntheticLaneChangeScenario(dt=0.2, max_steps=5)

    initial_state = scenario.reset()
    next_state = scenario.step(ControlAction(acceleration=0.0, steering=0.0))

    assert initial_state.time_step == 0
    assert next_state.time_step == 1
    assert next_state.ego.x > initial_state.ego.x
    assert len(next_state.agents) == 1
    assert next_state.metadata["scenario"] == "synthetic_lane_change"
