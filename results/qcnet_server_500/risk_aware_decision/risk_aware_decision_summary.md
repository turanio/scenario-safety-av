# QCNet Risk-Aware Decision Analysis

## Scope

This Stage C extension uses the existing 500 QCNet artifacts only; QCNet inference was not rerun. It remains open-loop point-trajectory screening on an AV2 validation subset, not closed-loop validation or collision-risk estimation.

## 1. Prediction-distribution risk proxies

For mode weights `p_k`, mode minimum center distances `d_k`, and `d_safe = 3.0 m`:

- `M_unsafe = sum p_k I(d_k < d_safe)` is the predicted mode mass in the unsafe screening region.
- `R_expected = sum p_k max(0, (d_safe - d_k) / d_safe)` combines mode mass with normalized threshold deficit.

Neither score is a calibrated collision probability or physical collision severity.

## 2. Intervention policies

The risk-mass policy brakes when `M_unsafe >= rho`; the expected-loss policy brakes when `R_expected >= eta`. The parameters `rho` and `eta` are operating points that represent risk tolerance and intervention cost. No threshold is presented as universally optimal.

## 3. Realized AV2 outcome reference

The reference is `ground_truth_min_distance < 3.0 m`, observed in **8 / 500** scenarios. It describes the one recorded AV2 future, not complete ground-truth safety risk; non-realized QCNet alternatives may remain plausible.

### Existing policies

| Policy | Interventions | Rate | TP | FP / extra | FN | Recall | Precision | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-1 | 13 | 2.6% | 5 | 8 | 3 | 0.625 | 0.385 | 0.016 |
| Worst-case | 31 | 6.2% | 7 | 24 | 1 | 0.875 | 0.226 | 0.049 |
| Probability-aware theta=0.05 | 19 | 3.8% | 6 | 13 | 2 | 0.750 | 0.316 | 0.026 |

### Selected risk-mass operating points

| Policy | Interventions | Rate | TP | FP / extra | FN | Recall | Precision | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rho=0.01 | 24 | 4.8% | 6 | 18 | 2 | 0.750 | 0.250 | 0.037 |
| rho=0.05 | 20 | 4.0% | 6 | 14 | 2 | 0.750 | 0.300 | 0.028 |
| rho=0.10 | 18 | 3.6% | 6 | 12 | 2 | 0.750 | 0.333 | 0.024 |

### Selected expected-loss operating points

| Policy | Interventions | Rate | TP | FP / extra | FN | Recall | Precision | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eta=0.01 | 20 | 4.0% | 5 | 15 | 3 | 0.625 | 0.250 | 0.030 |
| eta=0.05 | 12 | 2.4% | 4 | 8 | 4 | 0.500 | 0.333 | 0.016 |
| eta=0.10 | 7 | 1.4% | 2 | 5 | 6 | 0.250 | 0.286 | 0.010 |

The full sweep CSVs include fixed interpretable thresholds and exact observed score transition values. The curves characterize a conservatism/realized-event-recall trade-off rather than selecting an optimum.

## Exploratory score quality

| Score | Brier | AUROC | AUPRC |
|---|---:|---:|---:|
| Unsafe probability mass | 0.015197 | 0.925813 | 0.516848 |
| Expected distance-deficit risk | n/a | 0.917175 | 0.268263 |

AUPRC is reported as non-interpolated average precision. With only 8 realized positives in 500 scenarios, these discrimination and reliability results are exploratory. No fitting or recalibration was performed, and they do not support strong calibration claims.

## Outputs

- `risk_aware_per_scenario.csv`: direct scores and source point-distance metrics.
- `risk_mass_policy_sweep.csv`: all `rho` operating points.
- `expected_loss_policy_sweep.csv`: all `eta` operating points.
- `existing_policy_realized_outcomes.csv`: top-1, worst-case, and probability-filter results.
- `risk_score_quality.csv`: exploratory score metrics and reliability bins.
- `risk_decision_tradeoff.png`: recall/intervention-rate curves.
- `unsafe_mass_reliability.png`: fixed-bin exploratory reliability view.
