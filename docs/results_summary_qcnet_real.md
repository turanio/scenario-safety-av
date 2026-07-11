# QCNet Open-Loop Point-Trajectory Safety Analysis on AV2 Validation Scenarios

## What Was Evaluated

This study evaluates 50 scenario artifacts from the Argoverse 2 (AV2) motion forecasting validation split. Each artifact contains six real multimodal trajectory predictions from the released QCNet AV2 checkpoint, their mode probabilities, the AV2 ego future trajectory, and the focal target actor's ground-truth future trajectory. All trajectories are represented in the AV2 global coordinate frame over a 60-step, 6.0 s future horizon sampled at 10 Hz.

The analysis measure is the time-aligned Euclidean point distance between the ego future trajectory and either a predicted target trajectory or the target ground truth. It reports the minimum distance for the highest-probability QCNet mode (top-1), the minimum across all six modes (worst-case multimodal), and the ground-truth minimum. A point distance below 3.0 m is treated as a near miss for this analysis. The separate 1.0 m threshold remains a point-distance screening convention and is not vehicle-footprint collision checking.

This is open-loop point-trajectory safety analysis using real QCNet multimodal predictions and AV2 ego trajectories. It is not closed-loop autonomous driving evaluation and does not report CARLA results. The simple planner comparison does not alter the recorded ego trajectory or simulate the consequences of braking.

## Artifact Generation

The released `QCNet_AV2.ckpt` checkpoint was loaded in the external QCNet environment and run on a 50-scenario subset of AV2 validation data. Inference used the focal actor selected by the AV2/QCNet evaluation mask. QCNet's six refined trajectory modes were transformed back to global AV2 coordinates, and softmax probabilities were retained for each mode.

For every scenario, the export paired the multimodal focal-actor predictions with the time-aligned AV2 ego and focal-actor ground-truth histories, futures, and validity masks. These arrays were saved as dependency-light NumPy artifacts under `results/qcnet_batch/artifacts/`. The ranking files were then produced by measuring valid-step distances for the top-1 mode, every alternative mode, and ground truth. The existing 50 artifacts were not regenerated for this analysis step.

## Batch Summary

Across the 50 exported scenarios:

- 1 scenario had a top-1 QCNet minimum distance below the 3.0 m near-miss threshold.
- 3 scenarios had a worst-case multimodal minimum distance below 3.0 m.
- 1 scenario had a ground-truth minimum distance below 3.0 m.
- No top-1 or ground-truth trajectory fell below the 1.0 m point-distance threshold.
- 2 worst-case multimodal trajectories fell below the 1.0 m point-distance threshold; this does not establish a collision.
- 2 scenarios were hidden-risk cases in which top-1 remained above 3.0 m while a lower-probability mode fell below 3.0 m.

These are threshold counts within a selected 50-scenario subset. They are not collision rates, dataset-wide frequencies, or evidence that QCNet improves safety.

## Selected Scenarios

| Scenario | Type | Top-1 probability | Top-1 min (m) | Worst-case min (m) | Ground truth min (m) | Multimodal gap (m) |
|---|---|---:|---:|---:|---:|---:|
| `001749f1-bc1c-47fb-a13f-9ab1f2c050a8` | Hidden risk | 0.265 | 3.317 | 0.448 | 3.406 | 2.869 |
| `0091bad9-e7b2-4c07-aa12-6b5fd03c63d2` | High-confidence close interaction | 0.912 | 3.181 | 3.136 | 3.228 | 0.045 |
| `00351569-255c-433e-b97b-e2a844d1b6e0` | Real near miss | 0.306 | 2.597 | 2.045 | 2.162 | 0.552 |

### Hidden Risk: 001749

The top-1 prediction remains just above the near-miss threshold, with a minimum distance of 3.317 m. A lower-probability mode reaches 0.448 m, producing a 2.869 m multimodal gap. Multimodal prediction exposes possible lower-probability risk, but ground truth remains safe at 3.406 m. This is evidence of divergent predicted futures, not an observed collision or proof that braking would improve the outcome.

### High-Confidence Close Interaction: 0091bad

Top-1, worst-case multimodal, and ground-truth minimum distances are closely aligned at 3.181 m, 3.136 m, and 3.228 m. Their minima occur at zero-based steps 16, 19, and 18, corresponding to 1.7-2.0 s into the future rather than the horizon endpoint. The top-1 probability is 0.912, and map-context review places the trajectories coherently in the same road corridor. This case provides a high-confidence example of a consistent close interaction above but near the 3.0 m threshold.

### Real Near Miss: 003515

This is the strongest real near-miss example in the selected batch because ground truth, top-1, and worst-case multimodal distances are all below the 3.0 m near-miss threshold. Their respective minimum distances are 2.162 m, 2.597 m, and 2.045 m. The agreement indicates that the close interaction is present in the recorded future as well as in the QCNet predictions. This point-distance result is not a vehicle-footprint collision determination.

### Retained Appendix Case: 0058ed

Scenario `0058ed53-93bf-42a7-9bba-6df3f6ce20f5` remains available as an appendix or sensitivity example of a large top-1 versus multimodal gap. It was removed from the headline set because its strongest risky point occurs at the final horizon timestep, making the sub-meter worst-case point distance more vulnerable to endpoint effects. Its existing figures and metrics have been retained. Scenario `0091bad` was selected instead because its close interaction occurs earlier and is more consistent across top-1, worst-case multimodal, and ground-truth trajectories.

## Approximate actor-envelope distance

The center-distance analysis is retained as the primary metric, but it does not account for actor size. As a robustness check, an approximate envelope-adjusted distance subtracts a circular radius for each actor from every center distance:

`adjusted_distance = center_distance - ego_radius - target_radius`

The screening calculation uses a 2.25 m radius for ego and a 2.25 m radius for the target, for a combined adjustment of 4.5 m. A negative adjusted value means the two assumed circles overlap at that timestep. It does not establish vehicle contact or a collision: the approximation omits oriented vehicle length and width, heading, shape, and exact footprint intersection.

Because the circular radius is intentionally conservative, the binary circular-overlap screen is used only as a coarse screening indicator; the main interpretation comes from the relative ordering between top-1, worst-case, and ground truth.

| Scenario | Top-1 adjusted min (m) | Worst-case adjusted min (m) | Ground-truth adjusted min (m) | Circular overlap screen |
|---|---:|---:|---:|---|
| `001749` | -1.183 | -4.052 | -1.094 | All three negative |
| `0091bad` | -1.319 | -1.364 | -1.272 | All three negative |
| `003515` | -1.903 | -2.455 | -2.338 | All three negative |

For `001749`, all three series become negative under the circular assumption, while the worst-case mode remains substantially more negative than top-1 and ground truth. The multimodal margin gap therefore persists, although the circular screen is too coarse to determine physical contact. For `0091bad`, the three adjusted minima remain closely aligned, supporting the interpretation of a consistent close interaction. For `003515`, all three adjusted minima are negative, so the near-miss interpretation remains robust to this simple size adjustment. These results are screening evidence only and do not convert the open-loop analysis into collision checking or closed-loop safety evaluation.

## Top-1 and Multimodal Planner Decisions

The illustrative planner rule brakes only when its input minimum distance is below 3.0 m. For scenario 001749, the top-1 planner does not brake while the conservative multimodal planner brakes. For scenario 0091bad, neither planner brakes because both predicted minima remain above 3.0 m, although top-1, worst-case multimodal, and ground truth all identify a close interaction near the threshold. Both planners brake for scenario 003515.

The selected cases motivate uncertainty-aware planning by showing both disagreement across plausible modes and agreement around a close interaction. They do not prove that uncertainty-aware planning improves safety. The analysis does not simulate the effects or costs of intervention, and probability calibration, mode plausibility, and the trade-off between risk sensitivity and unnecessary braking remain open questions.

## SafeIO-style safety filter comparison

A lightweight SafeIO-style comparison evaluates three prediction-to-planning interfaces: the highest-probability mode only, the worst case across all modes, and a probability-aware filter retaining modes with `p >= 0.05`. These remain open-loop threshold decisions over center-distance trajectories, not simulated braking responses or a full SafeIO implementation.

For `001749`, top-1 returns `NO_BRAKE`, while worst-case returns `BRAKE` because mode 5 reaches 0.448 m. That mode has probability `p=0.024045`, so the probability-aware filter excludes it and returns `NO_BRAKE`; its closest retained mode remains at 3.257 m. For `0091bad`, all three filters return `NO_BRAKE`. For `003515`, all three return `BRAKE`, and the probability-aware decision is supported by retained mode 5 (`p=0.176845`) rather than the extremely low-probability worst-case mode.

The comparison shows how probability-aware filtering reduces sensitivity to extremely low-probability modes relative to worst-case filtering. It evaluates decision sensitivity to multimodal prediction uncertainty and motivates closed-loop validation, but it does not prove safety improvement, collision avoidance, or the consequences of either braking or not braking.

## Limitations

- The analysis is open loop; planner actions do not affect future ego or target motion.
- The 50 scenarios are a subset of AV2 validation data and are not a representative safety benchmark.
- The primary metric measures trajectory-center points; the supplementary circular-envelope screen approximates actor size but not oriented vehicle geometry.
- The 2.25 m radii are fixed assumptions, so adjusted results are sensitive to the chosen radius and may be conservative for some actors.
- Each artifact evaluates one focal target actor against the recorded ego trajectory rather than jointly reasoning over every road user.
- Worst-case mode selection ignores probability magnitude after the modes have been generated.
- The 3.0 m and 1.0 m thresholds are simple point-distance conventions, not validated universal safety definitions.
- No CARLA bridge, closed-loop controller, or counterfactual braking response is evaluated here.

## Next Steps

The selected and candidate trajectories have now been inspected with AV2 map context and a circular actor-envelope screen. The analysis can next be expanded to a larger, documented AV2 sample with probability-threshold sensitivity sweeps and, later, upgraded to oriented vehicle-footprint geometry. A later closed-loop phase should connect prediction-derived risk to a controller in CARLA or another simulator, measure the consequences of intervention, and report safety and conservatism together. Those future results must remain separate from the open-loop evidence reported here.

## Generated Outputs

- `results/qcnet_batch/figures/distance_over_time_hidden_risk_001749.png`
- `results/qcnet_batch/figures/distance_over_time_high_confidence_close_0091bad.png`
- `results/qcnet_batch/figures/distance_over_time_real_near_miss_003515.png`
- `results/qcnet_batch/figures/adjusted_distance_over_time_hidden_risk_001749.png`
- `results/qcnet_batch/figures/adjusted_distance_over_time_high_confidence_close_0091bad.png`
- `results/qcnet_batch/figures/adjusted_distance_over_time_real_near_miss_003515.png`
- `results/qcnet_batch/scenario_validation/`
- `results/qcnet_batch/scenario_validation_candidates/`
- `results/qcnet_batch/qcnet_selected_scenarios_summary.csv`
- `results/qcnet_batch/qcnet_selected_scenarios_envelope_summary.csv`
- `results/qcnet_batch/qcnet_planner_decision_comparison.csv`
- `results/qcnet_batch/qcnet_safety_filter_comparison.csv`
- `results/qcnet_batch/qcnet_safety_filter_summary.csv`

Retained appendix/sensitivity output:

- `results/qcnet_batch/figures/distance_over_time_large_gap_0058ed.png`
