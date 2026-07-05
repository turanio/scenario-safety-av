import json
from pathlib import Path

from av_safety_eval.experiments.run_baseline import run_baseline


def test_baseline_smoke_run_writes_metrics(tmp_path: Path) -> None:
    result = run_baseline(output_root=tmp_path, make_plot=False)

    metrics_file = Path(result["metrics_file"])
    assert metrics_file.exists()
    saved = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert saved["experiment_name"] == "baseline_constant_velocity_synthetic"
    assert saved["predictor"] == "constant_velocity"
    assert saved["horizon_steps"] == 30
    assert isinstance(saved["near_miss"], bool)
    assert isinstance(saved["collision"], bool)
