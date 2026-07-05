# Scenario-Based Safety Evaluation of Autonomous Driving

MSc thesis implementation for:

**Scenario-Based Safety Evaluation of Autonomous Driving Under Multimodal Trajectory Prediction Uncertainty**

## Thesis Goal

This project builds a scenario-based evaluation framework for studying how trajectory prediction uncertainty affects autonomous vehicle planning safety. The contribution is the evaluation pipeline and analysis, not a new simulator, a new diffusion architecture, or a production autonomous driving stack.

The planned pipeline is:

```text
Scenario -> Predictor -> Uncertainty -> Planner -> Simulation -> Safety metrics
```

## Current Implementation Status

Implemented now:

- `src/` layout Python package importable as `av_safety_eval`
- Core dataclasses for states, trajectories, predictions, controls, scenarios, and metrics
- Constant Velocity and Constant Acceleration baseline predictors
- Synthetic lane-change-like scenario that does not require external data
- Basic safety metrics
- Baseline demo that writes JSON metrics and an optional trajectory figure
- Baseline experiment matrix with four deterministic synthetic scenarios
- Closed-loop baseline planning evaluation with per-step logs
- Planner comparison between naive and standard closed-loop planners
- Synthetic multimodal uncertainty comparison with an uncertainty-aware conservative planner
- Results analysis command for thesis-ready tables, plots, and Markdown summary
- Unit tests for predictors, metrics, scenario stepping, baseline smoke execution, and matrix outputs

Placeholders exist for later:

- highway-env adapter
- CARLA adapter
- cVMD/cVMDx diffusion predictor integration
- conformal uncertainty estimation
- SafeIO-style planning extensions

## Installation

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

With conda:

```bash
conda env create -f environment.yml
conda activate scenario-safety-av
pip install -e .
```

GPU, CARLA, highD, cVMD, and highway-env are not required for the baseline.

## Running Tests

```bash
pytest
```

## Running Baseline Demo

After installing the package in editable mode:

```bash
python -m av_safety_eval.experiments.run_baseline
```

The demo writes:

- `results/metrics/baseline_constant_velocity_synthetic.json`
- `results/figures/baseline_constant_velocity_synthetic.png`

## Running Baseline Matrix

Run all deterministic synthetic baseline scenarios:

```bash
python -m av_safety_eval.experiments.run_baseline_matrix
```

The matrix includes:

- `safe_following`
- `near_miss_lane_change`
- `collision_risk_cut_in`
- `no_interaction`

The runner writes one JSON file per scenario under `results/metrics/` and an aggregate CSV:

- `results/metrics/baseline_matrix_summary.csv`

## Running Closed-Loop Baseline

Run the deterministic planner/predictor loop on all synthetic scenarios:

```bash
python -m av_safety_eval.experiments.run_closed_loop_baseline
```

At each simulation step, the runner predicts the target future with `ConstantVelocityPredictor`, chooses an ego action with `StandardPlanner`, advances the scenario, and logs state/action/safety values.

The runner writes:

- `results/logs/closed_loop_<scenario>.csv`
- `results/metrics/closed_loop_<scenario>.json`
- `results/metrics/closed_loop_baseline_summary.csv`

## Running Planner Comparison

Compare naive maintain-speed planning against the standard risk-aware baseline:

```bash
python -m av_safety_eval.experiments.run_planner_comparison
```

The comparison runs `naive` and `standard` planners on all synthetic scenarios. It writes:

- `results/logs/planner_comparison_<planner>_<scenario>.csv`
- `results/metrics/planner_comparison_<planner>_<scenario>.json`
- `results/metrics/planner_comparison_summary.csv`

## Running Synthetic Uncertainty Comparison

Compare deterministic planning against a synthetic multimodal uncertainty-aware planner:

```bash
python -m av_safety_eval.experiments.run_uncertainty_planner_comparison
```

This controlled experiment uses ambiguity scenarios where the most likely future is safe but a lower-probability cut-in future is risky. It includes:

- `ambiguous_cut_in`
- `delayed_cut_in`

It compares:

- `StandardPlanner` + `ConstantVelocityPredictor`
- `ConservativeUncertaintyPlanner` + `SyntheticMultimodalPredictor`

The runner writes:

- `results/logs/uncertainty_comparison_<planner>_<scenario>.csv`
- `results/metrics/uncertainty_comparison_<planner>_<scenario>.json`
- `results/metrics/uncertainty_planner_comparison_summary.csv`

## Running Results Analysis

Generate summary tables, thesis-ready plots, derived metrics, and a Markdown results summary:

```bash
python -m av_safety_eval.experiments.analyze_results
```

The command writes:

- `results/tables/`
- `results/figures/`
- `results/analysis/analysis_manifest.json`
- `docs/results_summary.md`

The key thesis table is:

- `results/tables/key_findings_table.csv`

## Project Structure

```text
configs/                Experiment configuration files
data/                   Local data folders; raw datasets are not committed
docs/                   Architecture and experiment notes
notes/                  Planning notes and thesis decisions
results/                Generated metrics, figures, and logs
src/av_safety_eval/     Main Python package
tests/                  Unit and smoke tests
old_setup/              Archived previous implementation
```

## Technology Notes

- highway-env is the current prototype simulator direction, but it is optional at this stage.
- CARLA is intended for final validation after the framework and baseline are working.
- cVMD/cVMDx is the primary diffusion trajectory prediction candidate for later integration.
- MotionDiffuser is currently treated as a state-of-the-art literature reference, not the first implementation target.
- highD is the primary practical highway dataset candidate, but access may require a manual request and no highD files should be committed.
- Baseline work must stay CPU-friendly and runnable without datasets, GPU, CARLA, cVMD, or highway-env.


# First results

baseline synthetic lane change
min distance 1.4 m
near miss true
collision false
