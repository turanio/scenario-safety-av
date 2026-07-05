"""Logging helpers for scripts and experiments."""

from __future__ import annotations

import logging


def get_logger(name: str = "av_safety_eval", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger without adding duplicate handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    return logger
