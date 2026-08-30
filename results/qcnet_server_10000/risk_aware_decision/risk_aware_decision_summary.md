# QCNet Risk-Aware Decision Analysis

## Scope

This analysis post-processes 10000 exported QCNet artifacts. It remains open-loop point-trajectory screening on an AV2 validation subset, not closed-loop validation or collision-risk estimation.

## 1. Prediction-distribution risk proxies

For mode weights `p_k`, mode minimum center distances `d_k`, and `d_safe = 3.0 m`:

- `M_unsafe = sum p_k I(d_k < d_safe)` is the predicted mode mass in the unsafe screening region.
- `R_expected = sum p_k max(0, (d_safe - d_k) / d_safe)` combines mode mass with normalized threshold deficit.

Neither score is a calibrated collision probability or physical collision severity.

## 2. Intervention policies

The risk-mass policy brakes when `M_unsafe >= rho`; the expected-loss policy brakes when `R_expected >= eta`. The parameters `rho` and `eta` are operating points that represent risk tolerance and intervention cost. No threshold is presented as universally optimal.

## 3. Realized AV2 outcome reference

The reference is `ground_truth_min_distance < 3.0 m`, observed in **159 / 10000** scenarios. It describes the one recorded AV2 future, not complete ground-truth safety risk; non-realized QCNet alternatives may remain plausible.

### Primary fixed-policy comparison

| Policy | Interventions | Rate | TP | FP / extra | FN | Recall | Precision | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-1 | 225 | 2.2% | 119 | 106 | 40 | 0.748 | 0.529 | 0.011 |
| Worst-case | 558 | 5.6% | 141 | 417 | 18 | 0.887 | 0.253 | 0.042 |
| Probability-aware theta=0.05 | 374 | 3.7% | 137 | 237 | 22 | 0.862 | 0.366 | 0.024 |
| Risk mass rho=0.10 | 341 | 3.4% | 133 | 208 | 26 | 0.836 | 0.390 | 0.021 |

### Selected risk-mass operating points

| Policy | Interventions | Rate | TP | FP / extra | FN | Recall | Precision | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rho=0.01 | 455 | 4.5% | 140 | 315 | 19 | 0.881 | 0.308 | 0.032 |
| rho=0.05 | 381 | 3.8% | 137 | 244 | 22 | 0.862 | 0.360 | 0.025 |
| rho=0.10 | 341 | 3.4% | 133 | 208 | 26 | 0.836 | 0.390 | 0.021 |

### Selected expected-loss operating points

| Policy | Interventions | Rate | TP | FP / extra | FN | Recall | Precision | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eta=0.01 | 337 | 3.4% | 123 | 214 | 36 | 0.774 | 0.365 | 0.022 |
| eta=0.05 | 198 | 2.0% | 89 | 109 | 70 | 0.560 | 0.449 | 0.011 |
| eta=0.10 | 114 | 1.1% | 43 | 71 | 116 | 0.270 | 0.377 | 0.007 |

The full sweep CSVs include fixed interpretable thresholds and exact observed score transition values. The curves characterize a conservatism/realized-event-recall trade-off rather than selecting an optimum.

## Exploratory score quality

| Score | Brier | AUROC | AUPRC |
|---|---:|---:|---:|
| Unsafe probability mass | 0.011465 | 0.936975 | 0.635297 |
| Expected distance-deficit risk | n/a | 0.930743 | 0.333565 |

AUPRC is reported as non-interpolated average precision and should be interpreted relative to the prevalence of 0.015900. No fitting or recalibration was performed. These metrics evaluate correspondence to the one recorded AV2 future reference, not calibrated physical collision probability.

## Outputs

- `risk_aware_per_scenario.csv`: direct scores and source point-distance metrics.
- `risk_mass_policy_sweep.csv`: all `rho` operating points.
- `expected_loss_policy_sweep.csv`: all `eta` operating points.
- `existing_policy_realized_outcomes.csv`: top-1, worst-case, and probability-filter results.
- `fixed_policy_comparison.csv`: the four predefined primary policies.
- `risk_score_quality.csv`: exploratory score metrics and reliability bins.
- `risk_decision_tradeoff.png`: recall/intervention-rate curves.
- `unsafe_mass_reliability.png`: adaptive-bin recorded-outcome reliability view.
