from pathlib import Path

import pytest

from av_safety_eval.common.types import AgentState
from av_safety_eval.predictors import CVMDAdapter
from av_safety_eval.predictors.cvmd_adapter import CVMDAdapter as DirectCVMDAdapter


def test_cvmd_adapter_imports_without_external_cvmd_dependency() -> None:
    assert CVMDAdapter is DirectCVMDAdapter


def test_cvmd_adapter_accepts_optional_paths() -> None:
    adapter = CVMDAdapter(model_path="models/cvmd.pt", config_path="configs/cvmd.yaml")

    assert adapter.model_path == Path("models/cvmd.pt")
    assert adapter.config_path == Path("configs/cvmd.yaml")


def test_cvmd_adapter_accepts_empty_paths() -> None:
    adapter = CVMDAdapter()

    assert adapter.model_path is None
    assert adapter.config_path is None


def test_cvmd_adapter_predict_raises_helpful_error() -> None:
    adapter = CVMDAdapter()
    history = [AgentState(agent_id="target", x=0.0, y=0.0, vx=1.0, vy=0.0)]

    with pytest.raises(NotImplementedError, match="cVMD/cVMDx integration is not implemented"):
        adapter.predict(history=history, horizon_steps=5, dt=0.2)
