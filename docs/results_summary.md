# Results Summary

## Available Experiments

The current repository contains synthetic, CPU-only experiments for open-loop prediction, closed-loop deterministic planning, planner comparison, and synthetic multimodal uncertainty-aware planning.

No diffusion model, cVMD/cVMDx integration, highD experiment, highway-env experiment, CARLA experiment, or GPU-based result is claimed here.

## Planner Comparison

In the `collision_risk_cut_in` scenario, naive planning leads to a collision, while the standard deterministic planner brakes and avoids it.

- Naive planner: min distance 0.203961 m, collision=true, interventions=0.
- Standard planner: min distance 4.434276 m, collision=false, interventions=7.

## Synthetic Multimodal Uncertainty Comparison

In the delayed cut-in case, the standard deterministic planner reacts to a single predicted future but still produces a near miss.
The uncertainty-aware conservative planner uses a synthetic multimodal predictor and reacts more conservatively to a plausible lower-probability cut-in future.

- Standard planner + Constant Velocity: min distance 2.335294 m, near_miss=true, interventions=12, first intervention at 1.2 s.
- Uncertainty-aware conservative planner + synthetic multimodal predictor: min distance 3.780212 m, near_miss=false, interventions=15, first intervention at 0.0 s.

## Interpretation

The results support the thesis logic in a controlled synthetic setting: considering multiple plausible futures can improve safety margins when a lower-probability behavior becomes safety-critical.
The explicit trade-off is conservatism: uncertainty-aware planning avoids the delayed cut-in near miss, but it brakes earlier and intervenes more often.

## Generated Artifacts

Tables:
- `results/tables/planner_comparison_table.csv`
- `results/tables/uncertainty_comparison_table.csv`
- `results/tables/delayed_cut_in_derived_metrics.csv`
- `results/tables/key_findings_table.csv`

Figures:
- `results/figures/planner_comparison_min_distance.png`
- `results/figures/planner_comparison_interventions.png`
- `results/figures/uncertainty_comparison_min_distance.png`
- `results/figures/delayed_cut_in_distance_over_time.png`
- `results/figures/delayed_cut_in_action_over_time.png`

## Limitations

These are controlled synthetic experiments. The synthetic multimodal predictor is not a diffusion model, and the uncertainty-aware conservative planner is not a full SafeIO implementation.
Future work should replace or extend the synthetic predictor with a validated diffusion-based predictor and evaluate on more realistic scenarios and datasets.
