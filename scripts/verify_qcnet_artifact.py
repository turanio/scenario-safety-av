"""Verify a QCNet `.npz` artifact and write a small JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from av_safety_eval.experiments.run_qcnet_artifact_evaluation import evaluate_qcnet_artifact
from av_safety_eval.predictors.qcnet_output_converter import load_qcnet_npz_prediction


def _to_point(values: np.ndarray) -> list[float]:
    return [round(float(values[0]), 6), round(float(values[1]), 6)]


def verify_qcnet_artifact(
    artifact_path: str | Path,
    run_evaluation: bool = True,
) -> dict[str, Any]:
    """Load a fake or real QCNet artifact and summarize its contents."""

    artifact_path = Path(artifact_path)
    prediction = load_qcnet_npz_prediction(artifact_path)
    scenario_id = getattr(prediction, "scenario_id", None)
    probabilities = prediction.probabilities or []

    modes = []
    for mode_index, trajectory in enumerate(prediction.trajectories):
        modes.append(
            {
                "mode_index": mode_index,
                "probability": round(float(probabilities[mode_index]), 6),
                "first_point": _to_point(trajectory.positions[0]),
                "last_point": _to_point(trajectory.positions[-1]),
            }
        )

    report: dict[str, Any] = {
        "artifact": str(artifact_path),
        "scenario_id": scenario_id,
        "target_actor_id": prediction.agent_id,
        "num_modes": len(prediction.trajectories),
        "horizon_steps": prediction.trajectories[0].steps,
        "dt": prediction.trajectories[0].dt,
        "probability_sum": round(float(sum(probabilities)), 6),
        "modes": modes,
    }
    if run_evaluation:
        report["offline_evaluation"] = evaluate_qcnet_artifact(artifact_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="QCNet .npz artifact path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qcnet_smoke/qcnet_real_artifact_verification.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only verify schema and trajectory summary.",
    )
    args = parser.parse_args()

    report = verify_qcnet_artifact(args.artifact, run_evaluation=not args.skip_evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("QCNet artifact verification complete")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
