# CARLA Validation Plan

Checked on 2026-07-05.

## Role of CARLA

CARLA should validate the planner and evaluation framework under controlled closed-loop scenarios. It is a validation simulator, not a training requirement for QCNet and not the thesis contribution by itself.

The minimum thesis version does not need to run QCNet directly inside CARLA. It can validate the same safety behavior already shown synthetically: deterministic planning reacts late, while uncertainty-aware conservative planning anticipates plausible risky futures.

## CARLA Practical Constraints

The CARLA documentation describes CARLA as a simulator with a Python API, actors, maps, sensors, traffic tools, and scenario control. The quick-start documentation recommends a dedicated GPU equivalent to an NVIDIA 2070 or better with at least 8 GB VRAM, about 20 GB of disk space, and a CARLA server process that Python clients connect to.

Do not add CARLA to the main project requirements yet. Keep CARLA behind the existing placeholder adapter until the simulator is installed and a small scenario can be run.

## Scenario Priority

1. Delayed cut-in.
2. Lane change.
3. Merge.
4. Crossing or intersection interaction if time allows.

## Minimum Validation

Recreate delayed cut-in in CARLA and compare:

- `StandardPlanner`;
- `ConservativeUncertaintyPlanner` / SafeIO-style uncertainty-aware planner;
- the same safety thresholds used in synthetic experiments.

Metrics:

- minimum distance;
- time-to-collision;
- near miss;
- collision;
- intervention count;
- first intervention time;
- success.

This validation can use synthetic multimodal predictions inside CARLA if QCNet-CARLA integration is not ready.

## Stronger Validation

Use QCNet-generated or QCNet-inspired multimodal predictions inside selected CARLA scenarios:

```text
AV2/QCNet scenario pattern -> CARLA controlled scenario -> planner comparison
```

The stronger version should show that the same planner design handles a more realistic simulator loop, not just a synthetic point-mass scenario.

## Direct QCNet-CARLA Integration

Directly running QCNet online inside CARLA is feasible in principle but risky for this thesis timeline. The main challenges are:

- QCNet is trained on AV2 distributions, maps, actor taxonomies, and coordinate frames;
- CARLA maps and actors are not AV2 scenarios;
- online inference latency may be high;
- converting CARLA live actor state into QCNet's expected graph/map input is non-trivial;
- validating coordinate transforms is time-consuming.

Therefore, direct QCNet-CARLA integration should be a stretch goal, not the minimum plan.

## Fallback

If QCNet-CARLA integration is too risky:

1. Use QCNet offline on AV2 to demonstrate real multimodal prediction conversion.
2. Use CARLA controlled scenarios to validate planner/evaluation behavior.
3. Use synthetic or QCNet-inspired multimodal futures in the CARLA planner loop.

This still supports the thesis claim that uncertainty-aware planning improves scenario safety margins, while honestly separating offline real-model prediction from closed-loop simulator validation.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| CARLA install/runtime issues | Medium/High | Keep synthetic results as baseline; use packaged release; avoid custom maps first |
| Scenario scripting takes too long | Medium | Start with delayed cut-in only |
| Direct QCNet input conversion is hard | High | Use offline QCNet and scenario-level CARLA recreation |
| GPU unavailable | Medium/High | Run synthetic and offline CPU smoke tests; request GPU for CARLA and QCNet |
| Metrics mismatch | Medium | Reuse existing metric definitions and log schema |

## Recommended First CARLA Milestone

Create one deterministic delayed cut-in scenario in CARLA and record per-step ego/target state, action, distance, near-miss, and collision fields in the same CSV style as the current closed-loop experiments.
