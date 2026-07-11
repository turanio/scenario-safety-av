# SafeIO-Style Safety Filter Results

## Motivation

Multimodal trajectory prediction creates an interface question for planning: should a safety decision use only the most probable future, every predicted future, or a probability-screened subset? This experiment compares those interfaces on the three selected QCNet scenarios. It evaluates decision sensitivity to multimodal prediction uncertainty; it does not simulate the effect of braking.

## Scope and Framing

This is a lightweight SafeIO-style safety filter, not a full SafeIO implementation. It applies transparent threshold rules to QCNet mode distances and probabilities. It does not reproduce a complete SafeIO optimization or control architecture, provide formal safety guarantees, or execute a closed-loop controller.

The experiment uses open-loop center distances between each predicted target trajectory and the recorded AV2 ego future. A policy returns `BRAKE` when the minimum distance visible to that policy is strictly below 3.0 m and `NO_BRAKE` otherwise. The `is_safe` field means only that the policy-visible minimum does not violate this threshold. It is not a claim that the physical scenario is safe.

## Policy Definitions

| Policy | Modes evaluated | Decision rule |
|---|---|---|
| Top-1 | Highest-probability QCNet mode | Brake if its minimum center distance is below 3.0 m |
| Worst-case multimodal | All QCNet modes | Brake if any mode has a minimum center distance below 3.0 m |
| Probability-aware multimodal | Modes with `p >= 0.05` | Brake if any retained mode has a minimum center distance below 3.0 m |

If no mode meets the probability threshold, the probability-aware policy falls back to the top-1 mode and records that fallback in its reason. This fallback was not needed for the three selected artifacts.

## Selected Scenario Results

| Scenario | Top-1 | Worst-case | Probability-aware (`p >= 0.05`) |
|---|---|---|---|
| `001749` Hidden risk | `NO_BRAKE`, 3.317 m | `BRAKE`, mode 5, 0.448 m | `NO_BRAKE`, 3.257 m |
| `0091bad` High-confidence close interaction | `NO_BRAKE`, 3.181 m | `NO_BRAKE`, 3.136 m | `NO_BRAKE`, 3.181 m |
| `003515` Real near miss | `BRAKE`, mode 3, 2.597 m | `BRAKE`, mode 0, 2.045 m | `BRAKE`, mode 5, 2.527 m |

### Hidden Risk: 001749

The top-1 filter evaluates mode 1 (`p=0.264896`) and does not brake. The worst-case filter brakes because mode 5 reaches 0.448 m, but that mode has probability `p=0.024045`. The probability-aware filter excludes mode 5 at the `p >= 0.05` cutoff; its closest retained trajectory is mode 0 at 3.257 m, so it does not brake.

This case shows that the planning decision depends on how low-probability predictions are exposed to the filter. Probability-aware filtering reduces sensitivity to extremely low-probability modes compared with worst-case filtering. It does not establish that ignoring mode 5 is safer, and the recorded ground-truth minimum remains above the point-distance threshold.

### High-Confidence Close Interaction: 0091bad

All three filters return `NO_BRAKE`. The top-1 mode has probability `p=0.912132` and reaches 3.181 m. The worst-case minimum is 3.136 m from mode 0 (`p=0.000377`), while only the top-1 mode passes the probability cutoff. All policy-visible minima remain above 3.0 m, so the interface choice does not change the action for this scenario.

### Real Near Miss: 003515

All three filters return `BRAKE`. The top-1 filter triggers on mode 3 at 2.597 m. The worst-case filter triggers on mode 0 at 2.045 m, despite its low probability of `p=0.000969`. The probability-aware filter excludes that mode but still triggers on retained mode 5 (`p=0.176845`) at 2.527 m. The braking decision therefore remains stable after probability screening, although the triggering evidence changes.

## Interpretation

The comparison separates two forms of conservatism. Worst-case filtering reacts to every exported mode regardless of probability, while top-1 filtering discards all alternatives. The probability-aware policy occupies a middle position by retaining several plausible modes while screening out the lowest-probability tail. In these examples, that distinction changes the decision for `001749`, does not matter for `0091bad`, and preserves the braking decision for `003515` through a different mode.

These results motivate studying prediction-to-planning interfaces, but they do not prove a safety improvement, collision avoidance, or a preferable universal probability threshold.

## Limitations

- The analysis is open loop: `BRAKE` is a policy label, not a simulated intervention.
- The filter uses point-to-point center distances, not exact oriented vehicle footprints or collision geometry.
- The `p >= 0.05` cutoff is a simple analysis choice and has not been calibrated as a safety threshold.
- QCNet mode probabilities may not be sufficiently calibrated for direct risk interpretation.
- Only three selected scenarios are compared, so the results do not estimate dataset-level performance.
- The experiment does not measure missed hazards, unnecessary braking, comfort, progress, or downstream control behavior.
- This lightweight filter does not provide the mechanisms or guarantees of a full SafeIO implementation.

## Next Step Toward Closed-Loop Validation

The immediate extension is a probability-threshold sensitivity sweep over a larger documented AV2 sample, reporting intervention frequency alongside ground-truth threshold outcomes. A later closed-loop experiment can connect the same policy interfaces to a controller in CARLA or another simulator and measure the consequences of braking. That phase should evaluate safety and conservatism together and remain clearly separated from the open-loop evidence reported here.

## Outputs

- `results/qcnet_batch/qcnet_safety_filter_comparison.csv`
- `results/qcnet_batch/qcnet_safety_filter_summary.csv`
