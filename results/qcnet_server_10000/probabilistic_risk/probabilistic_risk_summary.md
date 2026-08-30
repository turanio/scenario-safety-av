# QCNet Probabilistic-Risk Proxy Analysis

## Scope and definitions

This report post-processes a 10000-scenario QCNet/Argoverse 2 validation subset. For each predicted mode, the minimum center-to-center ego/target distance is computed over jointly valid future timesteps using the 3.0 m screening threshold.

`unsafe_probability_mass` is the sum of QCNet mode weights whose minimum distance is below 3.0 m. `expected_distance_deficit_risk` is the probability-weighted normalized threshold deficit. Both are risk proxies: QCNet probabilities are not safety-calibrated collision probabilities, and the deficit is not physical expected collision severity.

## Cohort integrity checks

Manifest/artifact counts and IDs, six-mode structure, probabilities, and jointly valid future horizons passed validation. No historical event-count fingerprint was requested for this cohort.

| Check | Observed |
|---|---:|
| Scenarios | 10000 |
| Worst-case threshold events | 558 |
| Top-1 threshold events | 225 |
| Recorded-ground-truth threshold events | 159 |
| Hidden-risk cases | 333 |

Risk-positive scenarios: **558 / 10000**.

## Unsafe probability mass

| Population | Mean | Median | P25 | P75 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 10000 scenarios | 0.023849 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.002022 | 1.000000 |
| Risk-positive only | 0.427405 | 0.244998 | 0.022464 | 0.949438 | 1.000000 | 1.000000 | 1.000000 |

## Expected normalized distance-deficit risk

| Population | Mean | Median | P25 | P75 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 10000 scenarios | 0.004324 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000218 | 0.113189 |
| Risk-positive only | 0.077487 | 0.023167 | 0.002438 | 0.079375 | 0.202019 | 0.366400 | 0.785881 |

## Probability-threshold retention

When no mode reaches a probability threshold, the highest-probability mode is retained as the same top-1 fallback used by the existing probability-aware filter.

| Threshold | Triggered | Fallback | Mean modes | Mean retained mass | Unsafe mass retained | Expected deficit retained |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 558 | 0 | 6.000 | 1.000000 | 100.00% | 100.00% |
| 0.001 | 515 | 0 | 5.560 | 0.999896 | 99.98% | 99.97% |
| 0.010 | 451 | 0 | 5.052 | 0.997574 | 99.63% | 99.36% |
| 0.030 | 398 | 0 | 4.550 | 0.988005 | 98.35% | 97.19% |
| 0.050 | 374 | 0 | 4.194 | 0.973952 | 96.52% | 94.11% |
| 0.100 | 330 | 0 | 3.470 | 0.920335 | 90.85% | 88.58% |
| 0.200 | 282 | 18 | 2.239 | 0.735924 | 71.54% | 69.24% |
| 0.300 | 237 | 2399 | 1.268 | 0.499344 | 47.11% | 45.98% |
| 0.500 | 225 | 8040 | 1.000 | 0.406952 | 38.82% | 38.18% |

## Top 15 by expected normalized deficit

| Rank | Scenario | Unsafe modes | Unsafe mass | Expected deficit | Top-1 min (m) | Worst min (m) | GT min (m) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `50977085-f57b-4243-929c-1a5f7d9284a4` | 6 | 1.000000 | 0.905563 | 0.261 | 0.205 | 8.358 |
| 2 | `f27a1d6e-1f38-44f2-8032-7b27939a3a0d` | 6 | 1.000000 | 0.897857 | 0.268 | 0.140 | 9.035 |
| 3 | `bea418f1-98c7-4781-8649-568ee340a7bc` | 4 | 0.999429 | 0.885488 | 0.271 | 0.271 | 6.460 |
| 4 | `1d483724-b370-4b6a-a8e1-0b7658fa1324` | 5 | 0.986804 | 0.876131 | 0.372 | 0.166 | 7.888 |
| 5 | `14c24cff-9175-41d7-b86d-8a6201c5c1b6` | 4 | 0.905295 | 0.860609 | 0.113 | 0.110 | 7.589 |
| 6 | `590bf360-21dd-4940-ad79-2b82f4379ae5` | 6 | 1.000000 | 0.793017 | 1.011 | 0.143 | 1.412 |
| 7 | `783a96ca-4601-4497-91d1-6b049ead1ef7` | 4 | 0.850503 | 0.780498 | 0.123 | 0.068 | 12.713 |
| 8 | `83b65b90-5ee7-4a3b-85b7-36dbaf36c4be` | 3 | 0.798827 | 0.759662 | 0.156 | 0.117 | 4.995 |
| 9 | `e4a51656-5d7a-408c-9034-2357eaf0b0c7` | 6 | 1.000000 | 0.747129 | 0.624 | 0.122 | 1.535 |
| 10 | `c71b9b93-ecaf-4a47-959b-d4fc369d8535` | 3 | 0.734596 | 0.706461 | 0.130 | 0.025 | 7.451 |
| 11 | `bd8c8d05-432e-4018-b790-69beb42e8c2c` | 4 | 0.803973 | 0.699492 | 0.127 | 0.097 | 5.423 |
| 12 | `b45309c7-13e1-4a50-9b2a-7751bb417f10` | 5 | 0.991485 | 0.698394 | 0.700 | 0.687 | 4.642 |
| 13 | `0f34e046-e12f-4e47-b02e-6cdaa3af3ec4` | 6 | 1.000000 | 0.662785 | 1.084 | 0.580 | 3.586 |
| 14 | `2dc03d85-09a6-4d5d-800e-a3d19d2bb5aa` | 2 | 0.936235 | 0.585196 | 0.793 | 0.793 | 17.017 |
| 15 | `f086dddb-3dc2-4384-aa43-7e25f2e4d457` | 3 | 0.675387 | 0.580253 | 0.261 | 0.261 | 6.317 |

## Key scenario checks

| Scenario | Top-1 probability | Worst-mode probability | Top-1 min (m) | Worst min (m) | GT min (m) | Unsafe mass | Severity proxy |
|---|---:|---:|---:|---:|---:|---:|---:|

## Factual interpretation

The two proxies distinguish low-probability severe alternatives from cases where more probability mass lies below the point-distance threshold. Probability filtering reduces the retained proxy totals as the cutoff rises, quantifying the trade-off already visible in the policy event counts. This remains open-loop point-trajectory screening; it does not establish calibrated collision risk, exact vehicle overlap, collision avoidance, or closed-loop safety improvement.
