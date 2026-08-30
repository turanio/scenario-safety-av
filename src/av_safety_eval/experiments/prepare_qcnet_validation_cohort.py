"""Create a deterministic AV2 validation cohort excluding a prior manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence


def read_manifest(path: Path) -> list[str]:
    scenario_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError(f"Manifest contains duplicate scenario IDs: {path}")
    return scenario_ids


def enumerate_scenario_ids(raw_root: Path) -> list[str]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"AV2 raw scenario directory not found: {raw_root}")
    scenario_ids = sorted(path.name for path in raw_root.iterdir() if path.is_dir())
    if len(scenario_ids) != len(set(scenario_ids)):
        raise RuntimeError("Available AV2 scenario IDs are not unique")
    return scenario_ids


def select_independent_cohort(
    available_ids: Sequence[str],
    historical_ids: Iterable[str],
    selected_count: int,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    """Sample without replacement, then sort for deterministic processing order."""
    if selected_count <= 0:
        raise ValueError("selected_count must be positive")
    available = list(available_ids)
    if len(available) != len(set(available)):
        raise ValueError("available_ids must be unique")
    historical = list(historical_ids)
    if len(historical) != len(set(historical)):
        raise ValueError("historical_ids must be unique")

    available_set = set(available)
    historical_set = set(historical)
    candidates = sorted(available_set - historical_set)
    if len(candidates) < selected_count:
        raise ValueError(
            f"Only {len(candidates)} independent candidates remain; "
            f"cannot select {selected_count}"
        )

    selected = sorted(random.Random(seed).sample(candidates, selected_count))
    overlap = set(selected) & historical_set
    if len(selected) != selected_count or len(set(selected)) != selected_count or overlap:
        raise RuntimeError("Selected cohort failed count, uniqueness, or overlap validation")

    summary = {
        "total_available_ids": len(available),
        "historical_manifest_ids": len(historical),
        "historical_ids_present_in_available": len(historical_set & available_set),
        "remaining_candidate_ids": len(candidates),
        "random_seed": seed,
        "selected_count": len(selected),
        "unique_selected_count": len(set(selected)),
        "historical_overlap_count": len(overlap),
    }
    return selected, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--selected-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output_manifest}")
    if args.summary_json is not None and args.summary_json.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {args.summary_json}")

    available_ids = enumerate_scenario_ids(args.raw_root)
    historical_ids = read_manifest(args.historical_manifest)
    selected_ids, summary = select_independent_cohort(
        available_ids,
        historical_ids,
        selected_count=args.selected_count,
        seed=args.seed,
    )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Manifest saved to: {args.output_manifest}")


if __name__ == "__main__":
    main()
