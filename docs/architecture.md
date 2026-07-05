# Architecture

The thesis framework follows this flow:

```text
Scenario source
  -> Simulator / scenario interface
  -> Agent state history
  -> Trajectory predictor
  -> Prediction set
  -> Uncertainty estimator
  -> Planner
  -> Control action
  -> Scenario step
  -> Metrics and visualizations
```

The current implementation establishes the interfaces and a synthetic baseline path. Later work can replace the synthetic scenario with highway-env or CARLA adapters and replace deterministic predictors with diffusion-based predictors.
