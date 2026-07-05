"""Human-readable labels for thesis result artifacts."""

from __future__ import annotations

LABELS = {
    "standard": "Standard",
    "naive": "Naive",
    "uncertainty_aware_conservative": "Uncertainty-aware",
    "constant_velocity": "Constant Velocity",
    "synthetic_multimodal": "Synthetic Multimodal",
    "delayed_cut_in": "Delayed cut-in",
    "ambiguous_cut_in": "Ambiguous cut-in",
    "collision_risk_cut_in": "Collision-risk cut-in",
    "near_miss_lane_change": "Near-miss lane change",
    "safe_following": "Safe following",
    "no_interaction": "No interaction",
}


def readable_label(value: object) -> str:
    """Return a human-readable label for known result identifiers."""

    text = str(value)
    return LABELS.get(text, text.replace("_", " ").title())
