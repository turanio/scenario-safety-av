# QCNet Inference Plan

Checked on 2026-07-05.

## Goal

Use QCNet as the real multimodal prediction component while keeping the main thesis framework lightweight and deterministic by default.

The target interface remains:

```text
PredictionSet
  trajectories: list[Trajectory]
  probabilities: list[float] | None

planner-facing stacked shape:
  num_modes x horizon_steps x 2
```

## QCNet Output Representation

QCNet predicts multiple modes. The public model code defaults to `num_modes=6`. For Argoverse 2, the documented configuration uses `num_future_steps=60`, corresponding to 6 seconds at 10 Hz.

The model produces refined trajectory positions and mode logits `pi`. During test, QCNet converts local predictions back into the global AV2 coordinate frame using the actor's last observed position and heading, then applies softmax to `pi`. The resulting trajectories and probabilities can be mapped directly into `PredictionSet`.

## Coordinate Frames

QCNet internally builds targets in an agent-centric frame. In test mode, it rotates and translates predictions back to global map coordinates.

The first integration should use global QCNet outputs rather than reproducing the internal transform. The conversion into this repository should preserve:

- x/y positions in meters;
- horizon order;
- `dt = 0.1` seconds for AV2;
- one `Trajectory` per QCNet mode;
- probabilities from softmaxed mode scores.

## Target and Ego Selection

Offline AV2 evaluation should start with the `FOCAL_TRACK` or `SCORED_TRACK` actor that QCNet evaluates. For planning experiments, choose:

- ego actor: the AV if available in the scenario representation, or the controlled vehicle chosen for the scenario extraction;
- target actor: the scored/focal actor whose future creates a safety-relevant interaction.

This mapping must be documented per scenario. Do not silently assume that AV2 focal actor is always the ego vehicle.

## Conversion Algorithm

For one evaluated actor:

1. Receive or load QCNet global predictions with shape `num_modes x horizon_steps x 2`.
2. Receive or derive mode probabilities with length `num_modes`.
3. Normalize probabilities if needed.
4. Create one `Trajectory(agent_id=target_id, positions=mode_positions, dt=0.1)` per mode.
5. Return `PredictionSet(agent_id=target_id, trajectories=trajectories, probabilities=probabilities)`.

If a future QCNet path produces trajectories for multiple actors, convert one target actor first. Multi-actor prediction can be added later without changing the baseline thesis claim.

## Offline Before CARLA

The first QCNet stage should run offline on AV2:

```text
AV2 scenario -> QCNet -> PredictionSet -> existing metrics/planner comparison
```

This avoids conflating three hard problems at once: model setup, dataset conversion, and CARLA control.

## Smallest Smoke Test

The smallest useful smoke test is:

1. External QCNet checkout works.
2. Released AV2 checkpoint loads.
3. One AV2 validation scenario is read.
4. QCNet emits 6 modes and mode scores for one actor.
5. A small local script or notebook saves those arrays.
6. `av_safety_eval` converts the arrays to `PredictionSet`.
7. The existing uncertainty-aware planner can evaluate risk over those modes.

## Adapter Boundary

The repository now contains:

```text
src/av_safety_eval/predictors/qcnet_adapter.py
```

The adapter is intentionally a placeholder. It does not import QCNet or AV2 at module import time. Later implementation should keep all heavy imports lazy and optional.

## Success Criteria for the Next Stage

Before CARLA work starts, the QCNet path should demonstrate:

- one AV2 scenario loaded externally;
- one released QCNet checkpoint used successfully;
- multimodal predictions converted to `PredictionSet`;
- an offline comparison between top-1 planning and multimodal conservative planning.
