"""Inspect a local QCNet repository checkout without cloning or importing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_PATHS = [
    "README.md",
    "environment.yml",
    "train_qcnet.py",
    "val.py",
    "test.py",
    "predictors",
    "datasets",
]
SEARCH_TERMS = [
    "ArgoverseV2Dataset",
    "num_modes",
    "num_future_steps",
    "pi",
    "softmax",
    "QCNet_AV2",
]
CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth"}


def _empty_report(repo_path: Path) -> dict:
    return {
        "repo_path": str(repo_path),
        "exists": False,
        "expected_paths": {path: False for path in EXPECTED_PATHS},
        "possible_checkpoint_files": [],
        "term_hits": {term: [] for term in SEARCH_TERMS},
    }


def inspect_qcnet_repo(repo_path: Path) -> dict:
    """Return a lightweight inventory of a local QCNet-style repository."""

    repo_path = repo_path.expanduser().resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        return _empty_report(repo_path)

    report = {
        "repo_path": str(repo_path),
        "exists": True,
        "expected_paths": {
            relative_path: (repo_path / relative_path).exists()
            for relative_path in EXPECTED_PATHS
        },
        "possible_checkpoint_files": [],
        "term_hits": {term: [] for term in SEARCH_TERMS},
    }

    files = [path for path in repo_path.rglob("*") if path.is_file()]
    report["possible_checkpoint_files"] = [
        path.relative_to(repo_path).as_posix()
        for path in sorted(files)
        if path.suffix.lower() in CHECKPOINT_SUFFIXES
    ]

    text_files = [
        path
        for path in files
        if path.suffix.lower() in {".py", ".md", ".yml", ".yaml", ".txt"}
        and path.stat().st_size <= 1_000_000
    ]
    for path in text_files[:300]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative_path = path.relative_to(repo_path).as_posix()
        for term in SEARCH_TERMS:
            if term in content:
                report["term_hits"][term].append(relative_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-path",
        type=Path,
        required=True,
        help="Path to an existing local QCNet checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qcnet_smoke/qcnet_repo_inspection.json"),
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    report = inspect_qcnet_repo(args.repo_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["exists"]:
        print(f"QCNet repository not found at {Path(args.repo_path).expanduser().resolve()}")
    else:
        print(f"QCNet repository found at {report['repo_path']}")
    print(json.dumps(report, indent=2))
    print(f"Inspection report written to {args.output}")


if __name__ == "__main__":
    main()
