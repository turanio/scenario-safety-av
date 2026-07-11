"""Probability-aware safety filters for multimodal trajectory predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BRAKE = "BRAKE"
NO_BRAKE = "NO_BRAKE"
DEFAULT_SAFETY_THRESHOLD_M = 3.0
DEFAULT_PROBABILITY_THRESHOLD = 0.05


@dataclass(frozen=True)
class SafetyFilterResult:
    """Decision and supporting evidence from one open-loop safety filter."""

    policy_name: str
    action: str
    is_safe: bool
    trigger_mode: int | None
    trigger_probability: float | None
    min_distance: float
    threshold_m: float
    reason: str


def _validate_threshold(value: float, name: str, upper_bound: float | None = None) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    if upper_bound is not None and threshold > upper_bound:
        raise ValueError(f"{name} must be at most {upper_bound}.")
    return threshold


def _mode_minima(per_mode_distances: np.ndarray) -> np.ndarray:
    values = np.asarray(per_mode_distances, dtype=float)
    if values.ndim == 1:
        minima = values.copy()
    elif values.ndim == 2 and values.shape[1] > 0:
        minima = np.empty(values.shape[0], dtype=float)
        for mode_index, series in enumerate(values):
            finite = series[np.isfinite(series)]
            if finite.size == 0:
                raise ValueError(
                    f"Mode {mode_index} must contain at least one finite distance."
                )
            minima[mode_index] = float(np.min(finite))
    else:
        raise ValueError(
            "per_mode_distances must contain per-mode minima with shape (modes,) "
            "or distance series with shape (modes, steps)."
        )

    if minima.size == 0:
        raise ValueError("per_mode_distances must contain at least one mode.")
    if not np.isfinite(minima).all():
        raise ValueError("Each mode minimum must be finite.")
    return minima


def _probabilities(mode_probabilities: np.ndarray, num_modes: int) -> np.ndarray:
    probabilities = np.asarray(mode_probabilities, dtype=float)
    if probabilities.ndim != 1 or probabilities.shape[0] != num_modes:
        raise ValueError("mode_probabilities must have shape (modes,).")
    if not np.isfinite(probabilities).all():
        raise ValueError("mode_probabilities must be finite.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("mode_probabilities must be between 0 and 1.")
    if not np.any(probabilities > 0.0):
        raise ValueError("At least one mode probability must be positive.")
    return probabilities


def _prepare_inputs(
    per_mode_distances: np.ndarray,
    mode_probabilities: np.ndarray,
    safety_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    minima = _mode_minima(per_mode_distances)
    probabilities = _probabilities(mode_probabilities, len(minima))
    threshold = _validate_threshold(safety_threshold_m, "safety_threshold_m")
    return minima, probabilities, threshold


def _result(
    policy_name: str,
    mode_index: int,
    minima: np.ndarray,
    probabilities: np.ndarray,
    threshold_m: float,
    brake_reason: str,
    no_brake_reason: str,
) -> SafetyFilterResult:
    min_distance = float(minima[mode_index])
    should_brake = min_distance < threshold_m
    return SafetyFilterResult(
        policy_name=policy_name,
        action=BRAKE if should_brake else NO_BRAKE,
        is_safe=not should_brake,
        trigger_mode=mode_index if should_brake else None,
        trigger_probability=(
            float(probabilities[mode_index]) if should_brake else None
        ),
        min_distance=min_distance,
        threshold_m=threshold_m,
        reason=brake_reason if should_brake else no_brake_reason,
    )


def evaluate_top1_filter(
    per_mode_distances: np.ndarray,
    mode_probabilities: np.ndarray,
    safety_threshold_m: float = DEFAULT_SAFETY_THRESHOLD_M,
) -> SafetyFilterResult:
    """Evaluate only the highest-probability trajectory mode."""

    minima, probabilities, threshold = _prepare_inputs(
        per_mode_distances, mode_probabilities, safety_threshold_m
    )
    mode_index = int(np.argmax(probabilities))
    probability = float(probabilities[mode_index])
    distance = float(minima[mode_index])
    return _result(
        policy_name="top1",
        mode_index=mode_index,
        minima=minima,
        probabilities=probabilities,
        threshold_m=threshold,
        brake_reason=(
            f"Top-1 mode {mode_index} (p={probability:.6f}) reaches "
            f"{distance:.6f} m, below the {threshold:.6f} m threshold."
        ),
        no_brake_reason=(
            f"Top-1 mode {mode_index} (p={probability:.6f}) remains at "
            f"{distance:.6f} m, at or above the {threshold:.6f} m threshold."
        ),
    )


def evaluate_worst_case_filter(
    per_mode_distances: np.ndarray,
    mode_probabilities: np.ndarray,
    safety_threshold_m: float = DEFAULT_SAFETY_THRESHOLD_M,
) -> SafetyFilterResult:
    """Evaluate the minimum distance across every trajectory mode."""

    minima, probabilities, threshold = _prepare_inputs(
        per_mode_distances, mode_probabilities, safety_threshold_m
    )
    mode_index = int(np.argmin(minima))
    probability = float(probabilities[mode_index])
    distance = float(minima[mode_index])
    return _result(
        policy_name="worst_case",
        mode_index=mode_index,
        minima=minima,
        probabilities=probabilities,
        threshold_m=threshold,
        brake_reason=(
            f"Mode {mode_index} (p={probability:.6f}) is the closest of all modes at "
            f"{distance:.6f} m, below the {threshold:.6f} m threshold."
        ),
        no_brake_reason=(
            f"All {len(minima)} modes remain at or above the {threshold:.6f} m "
            f"threshold; closest mode {mode_index} (p={probability:.6f}) reaches "
            f"{distance:.6f} m."
        ),
    )


def evaluate_probability_aware_filter(
    per_mode_distances: np.ndarray,
    mode_probabilities: np.ndarray,
    safety_threshold_m: float = DEFAULT_SAFETY_THRESHOLD_M,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> SafetyFilterResult:
    """Evaluate modes meeting a probability threshold, with top-1 fallback."""

    minima, probabilities, threshold = _prepare_inputs(
        per_mode_distances, mode_probabilities, safety_threshold_m
    )
    probability_threshold = _validate_threshold(
        probability_threshold, "probability_threshold", upper_bound=1.0
    )
    policy_name = f"probability_aware_p{round(probability_threshold * 100):03d}"
    eligible_modes = np.flatnonzero(probabilities >= probability_threshold)

    if eligible_modes.size == 0:
        mode_index = int(np.argmax(probabilities))
        probability = float(probabilities[mode_index])
        distance = float(minima[mode_index])
        fallback = (
            f"No mode meets p >= {probability_threshold:.6f}; falling back to "
            f"top-1 mode {mode_index} (p={probability:.6f})"
        )
        return _result(
            policy_name=policy_name,
            mode_index=mode_index,
            minima=minima,
            probabilities=probabilities,
            threshold_m=threshold,
            brake_reason=(
                f"{fallback}, which reaches {distance:.6f} m, below the "
                f"{threshold:.6f} m threshold."
            ),
            no_brake_reason=(
                f"{fallback}, which remains at {distance:.6f} m, at or above the "
                f"{threshold:.6f} m threshold."
            ),
        )

    eligible_minima = minima[eligible_modes]
    mode_index = int(eligible_modes[int(np.argmin(eligible_minima))])
    probability = float(probabilities[mode_index])
    distance = float(minima[mode_index])
    eligible_label = "mode" if len(eligible_modes) == 1 else "modes"
    remain_verb = "remains" if len(eligible_modes) == 1 else "remain"
    return _result(
        policy_name=policy_name,
        mode_index=mode_index,
        minima=minima,
        probabilities=probabilities,
        threshold_m=threshold,
        brake_reason=(
            f"Mode {mode_index} (p={probability:.6f}) is the closest of "
            f"{len(eligible_modes)} {eligible_label} with "
            f"p >= {probability_threshold:.6f}, "
            f"reaching {distance:.6f} m below the {threshold:.6f} m threshold."
        ),
        no_brake_reason=(
            f"All {len(eligible_modes)} eligible {eligible_label} with "
            f"p >= {probability_threshold:.6f} {remain_verb} at or above the "
            f"{threshold:.6f} m threshold; closest eligible mode {mode_index} "
            f"(p={probability:.6f}) reaches {distance:.6f} m."
        ),
    )
