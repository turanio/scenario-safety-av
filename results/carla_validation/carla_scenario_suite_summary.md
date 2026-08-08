# Controlled CARLA Scenario Suite

## Scope

This small suite is controlled closed-loop CARLA validation inspired by interaction patterns identified in the QCNet/Argoverse 2 analysis. It does not run QCNet online and is not intended as a general CARLA benchmark.

The QCNet/AV2 500-scenario evaluation provides the larger-scale open-loop evidence. These CARLA variants test whether differences between top-1, worst-case, and probability-aware policies affect closed-loop vehicle behavior under controlled conditions.

## Configuration

- CARLA endpoint: `127.0.0.1:2000`
- Synchronous mode: `true`
- Fixed delta: `0.05 s`
- Probability-aware cutoff: `p >= 0.05`
- Distance metric: ego-target center distance
- Future hypotheses: scripted; no online QCNet inference

## Results by Scenario

### `hidden_low_probability`

Inspired by: AV2 scenario 001749.

Mode probabilities (safe, moderate, aggressive): `0.93, 0.03, 0.04`.

Actual target behavior: `aggressive_cut_in`.

The aggressive cut-in hypothesis is below the probability cutoff, so the probability-aware policy is expected to behave closer to top-1 than to the worst-case policy.

| Policy | Minimum distance (m) | Near miss | Collision | Brake interventions | First brake (s) | Final speed (m/s) | Success |
|---|---:|---|---|---:|---:|---:|---|
| `top1_policy` | 4.609 | false | true | 5 | 1.25 | 6.144 | false |
| `worst_case_policy` | 7.079 | false | false | 7 | 0.10 | 8.743 | true |
| `probability_aware_policy` | 4.609 | false | true | 5 | 1.25 | 6.144 | false |

### `borderline_probability_aware`

Inspired by: AV2 scenario 00e2cd.

Mode probabilities (safe, moderate, aggressive): `0.82, 0.14, 0.04`.

Actual target behavior: `moderate_cut_in`.

The moderate cut-in hypothesis has probability 0.14 and passes the cutoff, so probability-aware and worst-case policies can respond before a top-1-only policy.

| Policy | Minimum distance (m) | Near miss | Collision | Brake interventions | First brake (s) | Final speed (m/s) | Success |
|---|---:|---|---|---:|---:|---:|---|
| `top1_policy` | 4.616 | false | true | 1 | 1.45 | 10.325 | false |
| `worst_case_policy` | 7.079 | false | false | 7 | 0.10 | 8.743 | true |
| `probability_aware_policy` | 7.079 | false | false | 7 | 0.10 | 8.743 | true |

### `near_miss_style`

Inspired by: AV2 scenario 003515.

Mode probabilities (safe, moderate, aggressive): `0.15, 0.70, 0.15`.

Actual target behavior: `moderate_cut_in`.

The moderate cut-in is the highest-probability hypothesis, so all three policies are expected to brake; timing and resulting vehicle behavior remain the comparison of interest.

| Policy | Minimum distance (m) | Near miss | Collision | Brake interventions | First brake (s) | Final speed (m/s) | Success |
|---|---:|---|---|---:|---:|---:|---|
| `top1_policy` | 7.079 | false | false | 7 | 0.10 | 8.743 | true |
| `worst_case_policy` | 7.079 | false | false | 7 | 0.10 | 8.743 | true |
| `probability_aware_policy` | 7.079 | false | false | 7 | 0.10 | 8.743 | true |

## CARLA Visual Evidence

RGB key-frame capture was disabled for this run.



## Limitations

The three variants are deliberately small and scripted. Minimum distance is center-to-center distance rather than oriented vehicle-footprint clearance. Collision is reported only when CARLA's collision sensor records an event.

The results can show policy-dependent closed-loop behavior in these controlled cases, but they do not prove collision avoidance, general safety improvement, or performance of online QCNet in CARLA.
