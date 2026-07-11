"""Distance-series helpers for open-loop trajectory safety screening."""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_NEAR_MISS_THRESHOLD_M = 3.0
DEFAULT_ENVELOPE_OVERLAP_THRESHOLD_M = 0.0


def compute_center_distances(
    ego_xy: np.ndarray,
    target_xy: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute aligned Euclidean center distances, using NaN for invalid steps."""

    ego = np.asarray(ego_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    if ego.ndim != 2 or ego.shape[1] != 2:
        raise ValueError("ego_xy must have shape (steps, 2).")
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target_xy must have shape (steps, 2).")
    if ego.shape != target.shape:
        raise ValueError("ego_xy and target_xy must have matching shapes.")

    if valid_mask is None:
        mask = np.ones(len(ego), dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != len(ego):
            raise ValueError("valid_mask must have shape (steps,).")

    finite_positions = np.isfinite(ego).all(axis=1) & np.isfinite(target).all(axis=1)
    distances = np.linalg.norm(ego - target, axis=1).astype(float)
    distances[~(mask & finite_positions)] = np.nan
    return distances


def compute_envelope_adjusted_distances(
    center_distances: np.ndarray,
    ego_radius_m: float = 2.25,
    target_radius_m: float = 2.25,
) -> np.ndarray:
    """Subtract circular actor radii from center distances.

    Negative values indicate approximate circular-envelope overlap, not confirmed
    collision geometry.
    """

    if not np.isfinite(ego_radius_m) or ego_radius_m < 0:
        raise ValueError("ego_radius_m must be finite and non-negative.")
    if not np.isfinite(target_radius_m) or target_radius_m < 0:
        raise ValueError("target_radius_m must be finite and non-negative.")

    distances = np.asarray(center_distances, dtype=float)
    if distances.ndim != 1:
        raise ValueError("center_distances must have shape (steps,).")
    return distances - ego_radius_m - target_radius_m


def summarize_distance_series(
    distances: np.ndarray,
    near_miss_threshold_m: float = DEFAULT_NEAR_MISS_THRESHOLD_M,
    envelope_overlap_threshold_m: float = DEFAULT_ENVELOPE_OVERLAP_THRESHOLD_M,
) -> dict[str, Any]:
    """Summarize the finite minimum and threshold screening outcomes."""

    values = np.asarray(distances, dtype=float)
    if values.ndim != 1:
        raise ValueError("distances must have shape (steps,).")
    if not np.isfinite(near_miss_threshold_m):
        raise ValueError("near_miss_threshold_m must be finite.")
    if not np.isfinite(envelope_overlap_threshold_m):
        raise ValueError("envelope_overlap_threshold_m must be finite.")

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return {
            "min_distance": float("nan"),
            "timestep_of_min": None,
            "below_near_miss_threshold": False,
            "below_collision_screening_threshold": False,
        }

    finite_values = np.where(finite_mask, values, np.nan)
    timestep = int(np.nanargmin(finite_values))
    min_distance = float(finite_values[timestep])
    return {
        "min_distance": min_distance,
        "timestep_of_min": timestep,
        "below_near_miss_threshold": min_distance < near_miss_threshold_m,
        "below_collision_screening_threshold": (
            min_distance < envelope_overlap_threshold_m
        ),
    }
