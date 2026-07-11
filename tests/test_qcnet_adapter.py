from pathlib import Path

import pytest

from av_safety_eval.common.types import AgentState
from av_safety_eval.predictors import QCNetAdapter
from av_safety_eval.predictors.qcnet_adapter import QCNetAdapter as DirectQCNetAdapter


def test_qcnet_adapter_imports_without_external_qcnet_dependency() -> None:
    assert QCNetAdapter is DirectQCNetAdapter


def test_qcnet_adapter_accepts_optional_paths_and_device() -> None:
    adapter = QCNetAdapter(
        checkpoint_path="models/qcnet.ckpt",
        config_path="configs/qcnet.yaml",
        device="cuda:0",
    )

    assert adapter.checkpoint_path == Path("models/qcnet.ckpt")
    assert adapter.config_path == Path("configs/qcnet.yaml")
    assert adapter.device == "cuda:0"


def test_qcnet_adapter_uses_cpu_by_default() -> None:
    adapter = QCNetAdapter()

    assert adapter.checkpoint_path is None
    assert adapter.config_path is None
    assert adapter.device == "cpu"


def test_qcnet_adapter_predict_raises_helpful_error() -> None:
    adapter = QCNetAdapter()
    history = [AgentState(agent_id="target", x=0.0, y=0.0, vx=1.0, vy=0.0)]

    with pytest.raises(NotImplementedError, match="QCNet integration is not implemented"):
        adapter.predict(history=history, horizon_steps=6, dt=0.1)
