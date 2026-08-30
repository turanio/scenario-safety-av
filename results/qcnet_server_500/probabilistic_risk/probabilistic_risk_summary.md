# QCNet Probabilistic-Risk Proxy Analysis

## Scope and definitions

This report post-processes the reproduced 500-scenario QCNet/Argoverse 2 validation subset. For each predicted mode, the minimum center-to-center ego/target distance is computed over jointly valid future timesteps using the 3.0 m screening threshold.

`unsafe_probability_mass` is the sum of QCNet mode weights whose minimum distance is below 3.0 m. `severity_weighted_risk` is the probability-weighted distance deficit below 3.0 m. Both are risk proxies: QCNet probabilities are not safety-calibrated collision probabilities, and the deficit is not physical expected collision severity.

## Reproduction sanity checks

All mandatory checks passed before these Stage C outputs were written.

| Check | Reproduced | Required |
|---|---:|---:|
| Scenarios | 500 | 500 |
| Worst-case threshold events | 31 | 31 |
| Top-1 threshold events | 13 | 13 |
| Recorded-ground-truth threshold events | 8 | 8 |
| Hidden-risk cases | 18 | 18 |

Risk-positive scenarios: **31 / 500**.

## Unsafe probability mass

| Population | Mean | Median | P25 | P75 | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| All 500 scenarios | 0.024825 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.004829 |
| Risk-positive only | 0.400403 | 0.135704 | 0.022765 | 0.917702 | 1.000000 | 1.000000 |

## Probability-weighted distance-deficit severity

| Population | Mean | Median | P25 | P75 | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| All 500 scenarios | 0.020445 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.005241 |
| Risk-positive only | 0.329752 | 0.070920 | 0.020179 | 0.262614 | 1.226140 | 1.633558 |

## Probability-threshold retention

When no mode reaches a probability threshold, the highest-probability mode is retained as the same top-1 fallback used by the existing probability-aware filter.

| Threshold | Triggered | Fallback | Mean modes | Mean retained mass | Unsafe mass retained | Severity retained |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 31 | 0 | 6.000 | 1.000000 | 100.00% | 100.00% |
| 0.001 | 29 | 0 | 5.462 | 0.999866 | 99.97% | 99.97% |
| 0.010 | 24 | 0 | 4.950 | 0.997691 | 99.51% | 99.38% |
| 0.030 | 20 | 0 | 4.508 | 0.989225 | 97.84% | 97.48% |
| 0.050 | 19 | 0 | 4.138 | 0.974420 | 95.87% | 95.85% |
| 0.100 | 14 | 0 | 3.424 | 0.921711 | 88.48% | 85.53% |
| 0.200 | 14 | 0 | 2.246 | 0.744114 | 76.66% | 80.71% |
| 0.300 | 13 | 110 | 1.256 | 0.502541 | 50.43% | 62.83% |
| 0.500 | 13 | 392 | 1.000 | 0.414125 | 39.81% | 36.93% |

## Top 15 by probability-weighted severity

| Rank | Scenario | Unsafe modes | Unsafe mass | Severity proxy | Top-1 min (m) | Worst min (m) | GT min (m) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `02ea59b1-3aac-46cf-b3f7-0358aa842499` | 4 | 0.981211 | 2.739127 | 0.177 | 0.177 | 5.190 |
| 2 | `0181eaca-74f6-4478-865f-df2d20802041` | 2 | 0.854192 | 1.960454 | 0.683 | 0.683 | 14.569 |
| 3 | `022c5148-8d15-4429-b649-6e7dc3354dc8` | 3 | 0.708938 | 1.306661 | 1.962 | 0.257 | 6.476 |
| 4 | `03b2fd2c-10a5-4e5d-a5a3-d2a941113b7e` | 6 | 1.000000 | 1.226140 | 1.621 | 1.621 | 2.705 |
| 5 | `03b8e752-e33e-46a0-a2c4-98e2ffe60edd` | 4 | 0.998257 | 0.395128 | 2.611 | 2.535 | 3.041 |
| 6 | `00351569-255c-433e-b97b-e2a844d1b6e0` | 6 | 1.000000 | 0.379535 | 2.597 | 2.045 | 2.162 |
| 7 | `0299fdda-3967-47f6-9cfa-0a57869df3ff` | 4 | 0.164717 | 0.355951 | 4.116 | 0.692 | 6.847 |
| 8 | `01ca1736-ec51-41aa-8c73-3338c574a83a` | 6 | 1.000000 | 0.267798 | 2.728 | 2.725 | 2.770 |
| 9 | `013563bf-4b77-464d-8930-612b2d58816c` | 1 | 0.707190 | 0.257430 | 2.636 | 2.636 | 3.351 |
| 10 | `01097a43-cccc-4ea1-97f7-c24bd10a8234` | 6 | 1.000000 | 0.214317 | 2.809 | 2.728 | 3.270 |
| 11 | `054a41cd-46ae-48e8-bb3f-3ac2eb951d66` | 5 | 0.988561 | 0.213500 | 2.693 | 2.693 | 2.871 |
| 12 | `03ea387c-b884-43e0-a6b4-071120cd08a2` | 1 | 0.057728 | 0.156320 | 6.717 | 0.292 | 8.601 |
| 13 | `01d69373-46c8-46bf-a500-aa7833faf886` | 3 | 0.440235 | 0.148066 | 2.696 | 2.547 | 12.561 |
| 14 | `04d21248-5ed5-49f3-bec3-fbf1c830e327` | 3 | 0.112830 | 0.134537 | 15.584 | 1.801 | 7.592 |
| 15 | `0494d287-7c6a-475c-aea1-594e45e4ac46` | 3 | 0.546128 | 0.089402 | 2.829 | 2.829 | 6.497 |

## Key scenario checks

| Scenario | Top-1 probability | Worst-mode probability | Top-1 min (m) | Worst min (m) | GT min (m) | Unsafe mass | Severity proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `001749` | 0.264890 | 0.024044950 | 3.317 | 0.448 | 3.406 | 0.024044950 | 0.061360370 |
| `00e2cd` | 0.301881 | 0.141278505 | 3.038 | 2.868 | 2.941 | 0.415135590 | 0.043086182 |
| `032618` | 0.371769 | 0.000007162 | 5.503 | 0.273 | 5.323 | 0.000007162 | 0.000019535 |

## Factual interpretation

The two proxies distinguish low-probability severe alternatives from cases where more probability mass lies below the point-distance threshold. Probability filtering reduces the retained proxy totals as the cutoff rises, quantifying the trade-off already visible in the policy event counts. This remains open-loop point-trajectory screening; it does not establish calibrated collision risk, exact vehicle overlap, collision avoidance, or closed-loop safety improvement.
