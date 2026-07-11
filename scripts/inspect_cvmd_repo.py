"""Inspect a local cVMD repository checkout without cloning or importing it."""

from __future__ import annotations

import argparse
from pathlib import Path


CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth", ".pkl"}
CONFIG_SUFFIXES = {".yaml", ".yml", ".json"}


def _relative_matches(repo_path: Path, patterns: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        for path in repo_path.rglob(pattern):
            if path.is_file():
                matches.append(path.relative_to(repo_path).as_posix())
    return sorted(set(matches))


def inspect_repo(repo_path: Path) -> dict[str, list[str] | bool | str]:
    """Return a lightweight inventory of a local cVMD-style repository."""

    repo_path = repo_path.expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    files = [path for path in repo_path.rglob("*") if path.is_file()]
    return {
        "repo_path": str(repo_path),
        "has_readme": any(path.name.lower().startswith("readme") for path in files),
        "has_license": any(path.name.lower().startswith("license") for path in files),
        "requirements_files": _relative_matches(
            repo_path,
            ("requirements*.txt", "environment*.yml", "environment*.yaml", "pyproject.toml"),
        ),
        "training_scripts": _relative_matches(repo_path, ("*train*.py", "main.py")),
        "inference_or_test_scripts": _relative_matches(repo_path, ("*infer*.py", "*test*.py")),
        "config_files": [
            path.relative_to(repo_path).as_posix()
            for path in sorted(files)
            if path.suffix.lower() in CONFIG_SUFFIXES
        ],
        "possible_checkpoint_files": [
            path.relative_to(repo_path).as_posix()
            for path in sorted(files)
            if path.suffix.lower() in CHECKPOINT_SUFFIXES
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-path",
        type=Path,
        required=True,
        help="Path to an existing local cVMD checkout.",
    )
    args = parser.parse_args()

    report = inspect_repo(args.repo_path)
    for key, value in report.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
