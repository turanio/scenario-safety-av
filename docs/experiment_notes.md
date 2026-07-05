# Experiment Notes

## Baseline Constant Velocity Synthetic

Purpose:

- Verify package structure.
- Verify deterministic trajectory prediction.
- Verify basic safety metrics.
- Produce the first repeatable metrics artifact without external data.

Run:

```bash
python -m av_safety_eval.experiments.run_baseline
```

Outputs:

- `results/metrics/baseline_constant_velocity_synthetic.json`
- `results/figures/baseline_constant_velocity_synthetic.png`

## Baseline Matrix

Purpose:

- Compare deterministic baseline behavior across a small set of synthetic interaction types.
- Produce one metrics JSON per scenario.
- Produce a compact CSV table suitable for thesis inspection and later plotting.

Run:

```bash
python -m av_safety_eval.experiments.run_baseline_matrix
```

Outputs:

- `results/metrics/baseline_constant_velocity_safe_following.json`
- `results/metrics/baseline_constant_velocity_near_miss_lane_change.json`
- `results/metrics/baseline_constant_velocity_collision_risk_cut_in.json`
- `results/metrics/baseline_constant_velocity_no_interaction.json`
- `results/metrics/baseline_matrix_summary.csv`

## Closed-Loop Baseline

Purpose:

- Evaluate the deterministic Constant Velocity predictor inside a repeated planning loop.
- Exercise `StandardPlanner` interventions under predicted near-miss and collision-risk futures.
- Produce step-by-step logs that can be inspected for action, distance, near-miss, and collision behavior.

Run:

```bash
python -m av_safety_eval.experiments.run_closed_loop_baseline
```

Outputs:

- `results/logs/closed_loop_safe_following.csv`
- `results/logs/closed_loop_near_miss_lane_change.csv`
- `results/logs/closed_loop_collision_risk_cut_in.csv`
- `results/logs/closed_loop_no_interaction.csv`
- `results/metrics/closed_loop_safe_following.json`
- `results/metrics/closed_loop_near_miss_lane_change.json`
- `results/metrics/closed_loop_collision_risk_cut_in.json`
- `results/metrics/closed_loop_no_interaction.json`
- `results/metrics/closed_loop_baseline_summary.csv`

## Planner Comparison

Purpose:

- Compare a naive maintain-speed planner against the deterministic risk-aware `StandardPlanner`.
- Show that planner choice changes safety outcomes under the same Constant Velocity predictor.
- Establish the interface needed for later uncertainty-aware planner comparison.

Run:

```bash
python -m av_safety_eval.experiments.run_planner_comparison
```

Outputs:

- `results/logs/planner_comparison_<planner>_<scenario>.csv`
- `results/metrics/planner_comparison_<planner>_<scenario>.json`
- `results/metrics/planner_comparison_summary.csv`

## Synthetic Multimodal Uncertainty Comparison

Purpose:

- Test the core thesis idea before integrating a real diffusion model.
- Create a controlled ambiguous scenario where the most likely target future is safe, while a lower-probability cut-in future is risky.
- Add a delayed cut-in scenario where the actual target behavior becomes risky after a short delay.
- Compare deterministic planning against a conservative uncertainty-aware planner that reacts to any plausible unsafe sampled future.

Run:

```bash
python -m av_safety_eval.experiments.run_uncertainty_planner_comparison
```

Outputs:

- `results/logs/uncertainty_comparison_standard_ambiguous_cut_in.csv`
- `results/logs/uncertainty_comparison_uncertainty_aware_conservative_ambiguous_cut_in.csv`
- `results/logs/uncertainty_comparison_standard_delayed_cut_in.csv`
- `results/logs/uncertainty_comparison_uncertainty_aware_conservative_delayed_cut_in.csv`
- `results/metrics/uncertainty_comparison_standard_ambiguous_cut_in.json`
- `results/metrics/uncertainty_comparison_uncertainty_aware_conservative_ambiguous_cut_in.json`
- `results/metrics/uncertainty_comparison_standard_delayed_cut_in.json`
- `results/metrics/uncertainty_comparison_uncertainty_aware_conservative_delayed_cut_in.json`
- `results/metrics/uncertainty_planner_comparison_summary.csv`

## Results Analysis

Purpose:

- Convert existing CSV and log outputs into thesis-ready tables, figures, derived metrics, and concise written interpretation.
- Keep claims limited to the current synthetic experiments.

Run:

```bash
python -m av_safety_eval.experiments.analyze_results
```

Outputs:

- `results/figures/planner_comparison_min_distance.png`
- `results/figures/planner_comparison_interventions.png`
- `results/figures/uncertainty_comparison_min_distance.png`
- `results/figures/delayed_cut_in_distance_over_time.png`
- `results/figures/delayed_cut_in_action_over_time.png`
- `results/tables/planner_comparison_table.csv`
- `results/tables/uncertainty_comparison_table.csv`
- `results/tables/delayed_cut_in_derived_metrics.csv`
- `results/tables/key_findings_table.csv`
- `docs/results_summary.md`
