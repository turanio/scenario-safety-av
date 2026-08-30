"""Validate a manifest-backed QCNet artifact cohort before downstream analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from av_safety_eval.experiments.analyze_qcnet_probabilistic_risk import (
    analyze_artifact,
    verify_cohort_integrity,
)
from av_safety_eval.experiments.prepare_qcnet_validation_cohort import read_manifest


DETAIL_FIELDS = (
    "artifact_file",
    "scenario_id",
    "target_actor_id",
    "num_modes",
    "prediction_steps",
    "ego_future_steps",
    "target_future_steps",
    "probability_sum",
    "joint_valid_steps",
    "status",
    "notes",
)


def historical_overlap(selected_ids: Sequence[str], historical_ids: Sequence[str]) -> set[str]:
    return set(selected_ids) & set(historical_ids)


def inspect_artifact_structure(
    path: Path,
    expected_num_modes: int = 6,
    expected_future_steps: int = 60,
    probability_sum_tolerance: float = 1e-4,
) -> dict[str, object]:
    """Inspect schema and trajectory shape without changing artifact contents."""
    with np.load(path, allow_pickle=False) as data:
        scenario_id = str(np.asarray(data["scenario_id"]).item())
        target_actor_id = str(np.asarray(data["target_actor_id"]).item())
        positions = np.asarray(data["positions"], dtype=float)
        probabilities = np.asarray(data["probabilities"], dtype=float).reshape(-1)
        ego_future = np.asarray(data["ego_future_positions"], dtype=float)
        target_future = np.asarray(data["target_future_positions"], dtype=float)
        ego_mask = np.asarray(data["ego_future_valid_mask"], dtype=bool)
        target_mask = np.asarray(data["target_future_valid_mask"], dtype=bool)

    errors = []
    expected_prediction_shape = (expected_num_modes, expected_future_steps, 2)
    if positions.shape != expected_prediction_shape:
        errors.append(f"positions shape {positions.shape} != {expected_prediction_shape}")
    if probabilities.shape != (expected_num_modes,):
        errors.append(f"probability shape {probabilities.shape}")
    if ego_future.shape != (expected_future_steps, 2):
        errors.append(f"ego future shape {ego_future.shape}")
    if target_future.shape != (expected_future_steps, 2):
        errors.append(f"target future shape {target_future.shape}")
    if ego_mask.shape != (expected_future_steps,):
        errors.append(f"ego validity-mask shape {ego_mask.shape}")
    if target_mask.shape != (expected_future_steps,):
        errors.append(f"target validity-mask shape {target_mask.shape}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        errors.append("probabilities are non-finite or negative")
    probability_sum = float(probabilities.sum())
    if not np.isclose(probability_sum, 1.0, atol=probability_sum_tolerance, rtol=0.0):
        errors.append(f"probability sum {probability_sum:.9f}")
    joint_valid_steps = int(np.sum(ego_mask & target_mask))
    if joint_valid_steps <= 0:
        errors.append("no jointly valid future timestep")

    return {
        "artifact_file": path.name,
        "scenario_id": scenario_id,
        "target_actor_id": target_actor_id,
        "num_modes": positions.shape[0] if positions.ndim >= 1 else 0,
        "prediction_steps": positions.shape[1] if positions.ndim >= 2 else 0,
        "ego_future_steps": ego_future.shape[0] if ego_future.ndim >= 1 else 0,
        "target_future_steps": target_future.shape[0] if target_future.ndim >= 1 else 0,
        "probability_sum": probability_sum,
        "joint_valid_steps": joint_valid_steps,
        "status": "pass" if not errors else "fail",
        "notes": "; ".join(errors),
    }


def read_export_summary_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [str(row["scenario_id"]) for row in csv.DictReader(handle)]


def validate_cohort(
    artifact_dir: Path,
    manifest_ids: Sequence[str],
    historical_ids: Sequence[str],
    expected_count: int,
    export_summary_csv: Path,
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    artifact_paths = sorted(path for path in artifact_dir.glob("*.npz") if path.is_file())
    details = []
    errors = []

    if len(manifest_ids) != expected_count:
        errors.append(f"manifest count {len(manifest_ids)} != {expected_count}")
    if len(set(manifest_ids)) != len(manifest_ids):
        errors.append("manifest IDs are not unique")

    overlap = historical_overlap(manifest_ids, historical_ids)
    if overlap:
        errors.append(f"historical overlap count is {len(overlap)}, expected 0")
    if len(artifact_paths) != expected_count:
        errors.append(f"artifact count {len(artifact_paths)} != {expected_count}")

    for path in artifact_paths:
        try:
            row = inspect_artifact_structure(path)
        except Exception as exc:
            row = {
                "artifact_file": path.name,
                "scenario_id": "",
                "target_actor_id": "",
                "num_modes": "",
                "prediction_steps": "",
                "ego_future_steps": "",
                "target_future_steps": "",
                "probability_sum": "",
                "joint_valid_steps": "",
                "status": "fail",
                "notes": str(exc),
            }
        details.append(row)
        if row["status"] != "pass":
            errors.append(f"{path.name}: {row['notes']}")

    artifact_ids = [str(row["scenario_id"]) for row in details if row["scenario_id"]]
    if len(set(artifact_ids)) != len(artifact_ids):
        errors.append("internal artifact scenario IDs are not unique")
    if set(artifact_ids) != set(manifest_ids):
        missing = sorted(set(manifest_ids) - set(artifact_ids))
        extra = sorted(set(artifact_ids) - set(manifest_ids))
        errors.append(f"artifact/manifest ID mismatch: missing={missing[:5]}, extra={extra[:5]}")
    if any(Path(str(row["artifact_file"])).stem != row["scenario_id"] for row in details):
        errors.append("one or more artifact filenames do not match internal scenario IDs")

    actor_keys = [
        (str(row["scenario_id"]), str(row["target_actor_id"]))
        for row in details
        if row["scenario_id"] and row["target_actor_id"]
    ]
    if len(actor_keys) != len(set(actor_keys)):
        errors.append("duplicate scenario/target-actor artifact pairs detected")

    if not export_summary_csv.is_file():
        errors.append(f"missing export summary: {export_summary_csv}")
        export_ids = []
    else:
        export_ids = read_export_summary_ids(export_summary_csv)
        if len(export_ids) != len(set(export_ids)):
            errors.append("export summary contains duplicate scenario IDs")
        if set(export_ids) != set(manifest_ids):
            errors.append("export summary IDs do not exactly equal manifest IDs")

    if not errors:
        records = [analyze_artifact(path) for path in artifact_paths]
        verify_cohort_integrity(records, manifest_ids)

    summary = {
        "expected_count": expected_count,
        "manifest_count": len(manifest_ids),
        "unique_manifest_ids": len(set(manifest_ids)),
        "artifact_count": len(artifact_paths),
        "unique_artifact_ids": len(set(artifact_ids)),
        "export_summary_count": len(export_ids),
        "historical_manifest_count": len(historical_ids),
        "historical_overlap_count": len(overlap),
        "six_mode_artifact_count": sum(row["num_modes"] == 6 for row in details),
        "sixty_step_artifact_count": sum(row["prediction_steps"] == 60 for row in details),
        "valid_joint_horizon_count": sum(
            isinstance(row["joint_valid_steps"], int) and row["joint_valid_steps"] > 0
            for row in details
        ),
        "failed_integrity_rows": sum(row["status"] != "pass" for row in details),
        "overall_status": "pass" if not errors else "fail",
    }
    return details, summary, errors


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: Mapping[str, object], errors: Sequence[str]) -> None:
    lines = [
        "# QCNet Artifact Cohort Integrity",
        "",
        "This gate validates artifact identity and structure before downstream quantitative analysis.",
        "",
        "| Check | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines.extend(["", "## Errors", ""])
    if errors:
        lines.extend(f"- {error}" for error in errors[:100])
        if len(errors) > 100:
            lines.append(f"- ... {len(errors) - 100} additional errors omitted")
    else:
        lines.append("No integrity errors detected.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-summary-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = (
        args.output_dir / "integrity_details.csv",
        args.output_dir / "integrity_summary.md",
        args.output_dir / "integrity_summary.json",
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite integrity outputs: {existing_outputs}")
    manifest_ids = read_manifest(args.manifest)
    historical_ids = read_manifest(args.historical_manifest)
    export_summary = args.export_summary_csv or args.artifact_dir / "batch_export_summary.csv"
    details, summary, errors = validate_cohort(
        args.artifact_dir,
        manifest_ids,
        historical_ids,
        args.expected_count,
        export_summary,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_paths[0], details)
    write_summary(output_paths[1], summary, errors)
    output_paths[2].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    if errors:
        raise RuntimeError(f"QCNet cohort integrity failed with {len(errors)} error(s)")


if __name__ == "__main__":
    main()
