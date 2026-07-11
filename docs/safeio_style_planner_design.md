# SafeIO-Style Uncertainty-Aware Planner Design

Checked on 2026-07-05.

## Scope

This document describes a SafeIO-style uncertainty-aware planner, not a full SafeIO implementation. The current implementation is intentionally conservative and simple. It is a decision/safety layer for evaluating how multimodal prediction uncertainty affects planning behavior.

## Core Logic

Given multiple predicted futures:

```text
for each plausible mode:
    compute minimum distance to ego rollout
    compute approximate time-to-collision when possible
    mark near-miss/collision risk using fixed thresholds
aggregate risk conservatively
choose an action that maintains safety margin
```

The planner should brake if any plausible mode violates the safety margin. It may maintain speed only when all considered modes stay outside the risk threshold.

## Use of QCNet Modes

QCNet provides multiple trajectory modes and mode scores. The planner should support two policies:

- all-modes policy: consider every QCNet mode;
- probability-threshold policy: consider modes whose probability is above a small threshold, for example 0.05.

For thesis clarity, the first proposed system should use a conservative threshold and report it. If thresholding hides a safety-critical low-probability mode, that trade-off must be discussed.

## Risk Metrics Per Mode

For each mode:

- predicted minimum distance;
- predicted time-to-collision if relative closing speed is meaningful;
- near-miss flag using the current near-miss threshold;
- collision flag using the current collision threshold;
- optional mode probability.

Aggregate values:

- worst-case minimum distance;
- minimum TTC across plausible modes;
- number or probability mass of risky modes;
- whether any plausible mode is unsafe.

## Difference From StandardPlanner

`StandardPlanner` reacts to one predicted future, currently from Constant Velocity or a top-1 forecast. It can miss lower-probability dangerous behavior when the most likely trajectory is safe.

The SafeIO-style uncertainty-aware planner reacts to the set of plausible futures. It is expected to intervene earlier and more often, especially when a lower-probability mode creates a cut-in, merge, or crossing conflict.

## Expected Trade-Off

Expected benefit:

- larger safety margin;
- fewer near misses or collisions in ambiguous interactions;
- clearer link between prediction uncertainty and planning safety.

Expected cost:

- more conservative braking;
- higher intervention count;
- possible comfort/efficiency loss;
- sensitivity to probability threshold and prediction calibration.

## Proposed Comparison

| System | Predictor | Planner |
|---|---|---|
| Baseline A | Constant Velocity | Naive |
| Baseline B | Constant Velocity | Standard |
| Baseline C | QCNet top-1 trajectory | Standard |
| Proposed | QCNet multimodal predictions | SafeIO-style uncertainty-aware planner |

## Evaluation

Use the existing metrics:

- minimum distance;
- time-to-collision;
- near miss;
- collision;
- intervention count;
- first intervention time;
- success.

Report both safety and conservatism. A scientifically honest result can say that the proposed planner improves safety margins but brakes earlier or more often.

## Implementation Boundary

The existing `ConservativeUncertaintyPlanner` is the first concrete SafeIO-style baseline. Future work can add:

- calibrated probability thresholds;
- risk-weighted action selection;
- comfort cost;
- multi-action rollout;
- formal SafeIO constraints if the actual algorithm is implemented.
