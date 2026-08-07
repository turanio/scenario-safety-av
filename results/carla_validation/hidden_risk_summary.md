# Controlled CARLA Hidden-Risk Validation

## Scope

This experiment is a controlled closed-loop CARLA validation of a cut-in pattern inspired by the QCNet/AV2 hidden-risk analysis. QCNet is not executed online in CARLA; all three future hypotheses are scripted and deterministic.

## Configuration

- CARLA endpoint: `127.0.0.1:2000`
- Synchronous mode: `true`
- Fixed delta: `0.05 s`
- Scenario duration: `6.0 s`
- Center-distance threshold: `3.0 m`
- Probability-aware cutoff: `p >= 0.05`
- Actual target behavior: scripted moderate cut-in
- Hypotheses: safe continuation, moderate cut-in, aggressive cut-in
- Mode probabilities: `0.78, 0.18, 0.04`
- The probability-aware policy includes the safe and moderate modes
- Braking interventions count transitions into a braking episode
- Scenario success means the rollout completed all expected steps; an available collision-sensor event makes it false

## Results

| Policy | Minimum distance (m) | Near miss | Collision sensor | Brake interventions | First brake (s) | Final ego speed (m/s) | Success |
|---|---:|---|---|---:|---:|---:|---|
| `top1_policy` | 4.616 | false | true | 1 | 1.45 | 10.325 | false |
| `worst_case_policy` | 7.079 | false | false | 7 | 0.10 | 8.743 | true |
| `probability_aware_policy` | 7.079 | false | false | 7 | 0.10 | 8.743 | true |

## Interpretation

The three rows compare how mode selection changes the timing and frequency of braking under identical initial conditions. Differences in minimum distance are outcomes of this scripted CARLA interaction; they do not establish a general safety improvement.

## Limitations

The forecast modes are synthetic and scenario-specific. The reported minimum distance is center-to-center distance, not oriented vehicle-footprint clearance. Collision is true only when the CARLA collision sensor records an event; if that sensor is unavailable, the CSV reports false with an explicit note.

This is one controlled closed-loop experiment, not evidence of collision avoidance, a full autonomous-driving stack, online QCNet integration, or general closed-loop safety improvement.
