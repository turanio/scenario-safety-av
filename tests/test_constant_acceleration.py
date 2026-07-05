import numpy as np
import pytest

from av_safety_eval.common.types import AgentState
from av_safety_eval.predictors.constant_acceleration import ConstantAccelerationPredictor


def test_constant_acceleration_prediction_output_shape_and_values() -> None:
    predictor = ConstantAccelerationPredictor()
    history = [
        AgentState(
            agent_id="target",
            x=0.0,
            y=0.0,
            vx=1.0,
            vy=0.0,
            ax=1.0,
            ay=0.0,
        )
    ]

    prediction = predictor.predict(history, horizon_steps=3, dt=1.0)

    trajectory = prediction.trajectories[0]
    assert trajectory.positions.shape == (3, 2)
    np.testing.assert_allclose(
        trajectory.positions,
        np.array(
            [
                [1.5, 0.0],
                [4.0, 0.0],
                [7.5, 0.0],
            ]
        ),
    )
    assert prediction.probabilities == [1.0]


@pytest.mark.parametrize(
    ("history", "horizon_steps", "dt", "message"),
    [
        ([], 3, 1.0, "history"),
        ([AgentState(agent_id="a", x=0.0, y=0.0, vx=0.0, vy=0.0)], -1, 1.0, "horizon_steps"),
        ([AgentState(agent_id="a", x=0.0, y=0.0, vx=0.0, vy=0.0)], 3, -0.1, "dt"),
    ],
)
def test_constant_acceleration_rejects_invalid_inputs(
    history: list[AgentState],
    horizon_steps: int,
    dt: float,
    message: str,
) -> None:
    predictor = ConstantAccelerationPredictor()
    with pytest.raises(ValueError, match=message):
        predictor.predict(history, horizon_steps=horizon_steps, dt=dt)
