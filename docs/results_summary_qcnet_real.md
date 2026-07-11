# QCNet Open-Loop Safety Analysis on AV2 Validation Scenarios

## What Was Evaluated

This study evaluates 50 scenario artifacts from the Argoverse 2 (AV2) motion forecasting validation split. Each artifact contains six real multimodal trajectory predictions from the released QCNet AV2 checkpoint, their mode probabilities, the AV2 ego future trajectory, and the focal target actor's ground-truth future trajectory. All trajectories are represented in the AV2 global coordinate frame over a 60-step, 6.0 s future horizon sampled at 10 Hz.

The safety measure is the time-aligned Euclidean distance between the ego future trajectory and either a predicted target trajectory or the target ground truth. The analysis reports the minimum distance for the highest-probability QCNet mode (top-1), the minimum across all six modes (worst-case multimodal), and the ground-truth minimum. A distance below 3.0 m is treated as a near miss and a distance below 1.0 m as a collision-threshold event.

This is open-loop safety analysis using real QCNet multimodal predictions and AV2 ego trajectories. It is not closed-loop autonomous driving evaluation and does not report CARLA results. The simple planner comparison does not alter the recorded ego trajectory or simulate the consequences of braking.

## Artifact Generation

The released `QCNet_AV2.ckpt` checkpoint was loaded in the external QCNet environment and run on a 50-scenario subset of AV2 validation data. Inference used the focal actor selected by the AV2/QCNet evaluation mask. QCNet's six refined trajectory modes were transformed back to global AV2 coordinates, and softmax probabilities were retained for each mode.

For every scenario, the export paired the multimodal focal-actor predictions with the time-aligned AV2 ego and focal-actor ground-truth histories, futures, and validity masks. These arrays were saved as dependency-light NumPy artifacts under `results/qcnet_batch/artifacts/`. The ranking files were then produced by measuring valid-step distances for the top-1 mode, every alternative mode, and ground truth. The existing 50 artifacts were not regenerated for this analysis step.

## Batch Summary

Across the 50 exported scenarios:

- 1 scenario had a top-1 QCNet minimum distance below the 3.0 m near-miss threshold.
- 3 scenarios had a worst-case multimodal minimum distance below 3.0 m.
- 1 scenario had a ground-truth minimum distance below 3.0 m.
- No top-1 or ground-truth trajectory fell below the 1.0 m collision threshold.
- 2 worst-case multimodal trajectories fell below 1.0 m.
- 2 scenarios were hidden-risk cases in which top-1 remained above 3.0 m while a lower-probability mode fell below 3.0 m.

These are threshold counts within a selected 50-scenario subset. They are not collision rates, dataset-wide frequencies, or evidence that QCNet improves safety.

## Selected Scenarios

| Scenario | Type | Top-1 probability | Top-1 min (m) | Worst-case min (m) | Ground truth min (m) | Multimodal gap (m) |
|---|---|---:|---:|---:|---:|---:|
| `001749f1-bc1c-47fb-a13f-9ab1f2c050a8` | Hidden risk | 0.265 | 3.317 | 0.448 | 3.406 | 2.869 |
| `0058ed53-93bf-42a7-9bba-6df3f6ce20f5` | Large top-1 vs multimodal gap | 0.294 | 8.173 | 0.646 | 9.283 | 7.527 |
| `00351569-255c-433e-b97b-e2a844d1b6e0` | Real near miss | 0.306 | 2.597 | 2.045 | 2.162 | 0.552 |

### Hidden Risk: 001749

The top-1 prediction remains just above the near-miss threshold, with a minimum distance of 3.317 m. A lower-probability mode reaches 0.448 m, producing a 2.869 m multimodal gap. Multimodal prediction exposes possible lower-probability risk, but ground truth remains safe at 3.406 m. This is evidence of divergent predicted futures, not an observed collision or proof that braking would improve the outcome.

### Large Gap: 0058ed

The top-1 prediction remains well separated from ego at 8.173 m, while the worst-case mode reaches 0.646 m. The resulting 7.527 m gap is a clear example of a risk signal that is absent from the highest-probability trajectory. Multimodal prediction exposes possible lower-probability risk, but ground truth remains safe at 9.283 m.

### Real Near Miss: 003515

This is the strongest real near-miss example in the selected batch because ground truth, top-1, and worst-case multimodal distances are all below the 3.0 m near-miss threshold. Their respective minimum distances are 2.162 m, 2.597 m, and 2.045 m. The agreement indicates that the close interaction is present in the recorded future as well as in the QCNet predictions, although no trajectory falls below the 1.0 m collision threshold.

## Top-1 and Multimodal Planner Decisions

The illustrative planner rule brakes when its input minimum distance is below 3.0 m. A top-1 planner therefore does not brake for scenarios 001749 or 0058ed, while a conservative multimodal planner brakes because at least one predicted mode crosses the threshold. Both planners brake for scenario 003515.

This comparison shows how considering lower-probability modes can change a threshold-based decision. It does not establish that the conservative decision is safer: in the first two scenarios the recorded ground truth remains above 3.0 m, and the analysis does not simulate the effects or costs of intervention. Probability calibration, mode plausibility, and the trade-off between risk sensitivity and unnecessary braking remain open questions.

## Limitations

- The analysis is open loop; planner actions do not affect future ego or target motion.
- The 50 scenarios are a subset of AV2 validation data and are not a representative safety benchmark.
- Distance is measured between trajectory points and does not account for actor dimensions, orientation, or collision geometry.
- Each artifact evaluates one focal target actor against the recorded ego trajectory rather than jointly reasoning over every road user.
- Worst-case mode selection ignores probability magnitude after the modes have been generated.
- The 3.0 m and 1.0 m thresholds are simple analysis conventions, not validated universal definitions of near miss and collision.
- No CARLA bridge, closed-loop controller, or counterfactual braking response is evaluated here.

## Next Steps

The immediate next step is to inspect the selected trajectories alongside map context and verify that the low-distance modes are behaviorally plausible. The analysis can then be expanded to a larger, documented AV2 sample and augmented with actor-envelope distances and probability-aware risk measures. A later closed-loop phase should connect prediction-derived risk to a controller in CARLA or another simulator, measure the consequences of intervention, and report safety and conservatism together. Those future results must remain separate from the open-loop evidence reported here.

## Generated Outputs

- `results/qcnet_batch/figures/distance_over_time_hidden_risk_001749.png`
- `results/qcnet_batch/figures/distance_over_time_large_gap_0058ed.png`
- `results/qcnet_batch/figures/distance_over_time_real_near_miss_003515.png`
- `results/qcnet_batch/qcnet_selected_scenarios_summary.csv`
- `results/qcnet_batch/qcnet_planner_decision_comparison.csv`
