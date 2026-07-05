"""Placeholder for future highD dataset integration."""

from __future__ import annotations

from pathlib import Path


class HighDPlaceholderDataset:
    """Documents the future highD integration boundary."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def load(self) -> None:
        """Raise until highD access and schema decisions are finalized."""

        raise NotImplementedError(
            "highD loading is intentionally not implemented yet. "
            "Use synthetic scenarios for the initial baseline."
        )
