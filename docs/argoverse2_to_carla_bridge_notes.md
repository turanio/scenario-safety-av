# Argoverse 2 to CARLA Bridge Notes

Checked on 2026-07-05.

## Principle

Argoverse 2 and CARLA are different environments. AV2 is a real-world recorded dataset with local HD maps and actor histories. CARLA is a controllable simulator with its own maps, actors, physics, and coordinate system.

Direct AV2 scene replay in CARLA should not be promised as the first integration step.

## Bridge Strategy

The first bridge should be scenario-level, not exact map-level replay.

Extract interaction patterns from AV2/QCNet:

- relative longitudinal and lateral positions;
- relative speeds;
- lateral movement direction and timing;
- approximate lane topology;
- interaction type;
- risk timing;
- predicted multimodal alternatives.

Then recreate those patterns in CARLA with controlled vehicles.

## Useful Pattern Types

- delayed cut-in;
- lane change close to ego;
- merge into ego lane;
- crossing or intersection conflict;
- close following with sudden deceleration;
- ambiguous actor intent where top-1 is safe but another plausible future is risky.

## Coordinate Considerations

AV2 positions are in local map coordinates. CARLA uses its own world coordinates and map-specific road topology. The bridge should convert interaction geometry, not raw global coordinates.

For each selected AV2 scenario, record a compact normalized description:

```text
ego_start_position
ego_speed
target_relative_position
target_speed
target_lateral_offset
lane_width
interaction_start_time
interaction_duration
target_behavior
```

This description can parameterize a CARLA scenario without requiring exact AV2 map replay.

## Relationship to QCNet

QCNet can be used offline to identify and characterize ambiguous multimodal interactions. CARLA can then validate whether the planner responds safely when a similar interaction unfolds in closed loop.

This separation is scientifically cleaner than claiming QCNet is trained or natively deployed in CARLA.

## Minimum Bridge

Use the current synthetic delayed cut-in parameters as the first CARLA scenario template. Then add AV2-inspired variations once QCNet predictions are available:

- earlier or later cut-in start;
- faster or slower target;
- smaller or larger lateral offset;
- different initial gap;
- different probability assigned to the risky mode.

## Success Criteria

The bridge is successful if it can show:

- the same scenario family in synthetic and CARLA settings;
- comparable metrics;
- standard planner reacts later or less conservatively;
- SafeIO-style uncertainty-aware planner preserves a larger safety margin;
- all assumptions about AV2-to-CARLA abstraction are documented.
