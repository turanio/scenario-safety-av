# QCNet Probability-Threshold Sensitivity Analysis

## Setup

This experiment evaluates how a probability-aware safety filter changes as the minimum accepted prediction-mode probability changes.

- Dataset: Argoverse 2 validation subset
- Sample size: 500 scenarios
- Predictor: released QCNet AV2 checkpoint
- Prediction modes: 6
- Safety threshold: 3.0 m
- Evaluation type: open-loop distance-based analysis

## Main Result

| Probability threshold | Brake cases | Hidden-risk detections | Missed worst-case brake cases | Mean eligible modes |
|---:|---:|---:|---:|---:|
| 0.000 | 31 | 18 | 0 | 6.000 |
| 0.001 | 29 | 16 | 2 | 5.462 |
| 0.010 | 24 | 11 | 7 | 4.950 |
| 0.030 | 20 | 7 | 11 | 4.508 |
| 0.050 | 19 | 6 | 12 | 4.138 |
| 0.100 | 14 | 1 | 17 | 3.424 |
| 0.200 | 14 | 1 | 17 | 2.246 |
| 0.300 | 13 | 0 | 18 | 1.256 |
| 0.500 | 13 | 0 | 18 | 1.000 |

## Interpretation

The sweep shows a clear safety-conservatism trade-off.

At p >= 0.000, all prediction modes are considered. This matches worst-case filtering and detects 31 brake cases, including 18 hidden-risk cases where top-1 alone remains above the safety threshold.

As the probability threshold increases, fewer modes are considered. This reduces the number of brake decisions, but it also causes more worst-case risky modes to be ignored.

At p >= 0.050, the filter brakes in 19 cases and still detects 6 hidden-risk cases. However, it misses 12 worst-case brake cases compared with considering all modes.

At p >= 0.300 and p >= 0.500, the filter behaves almost like top-1 filtering. It produces 13 brake cases and detects no hidden-risk cases.

## Fallback Caveat

For high probability thresholds, many scenarios have no mode above the selected cutoff. In these cases, the script falls back to the top-1 mode.

Fallback counts:

- p >= 0.300: 110 scenarios
- p >= 0.500: 392 scenarios

Therefore, the high-threshold results should be interpreted as top-1-dominated behaviour rather than fully multimodal probability-aware filtering.

## Thesis Relevance

This experiment supports the thesis claim that safety evaluation under multimodal prediction uncertainty depends strongly on how low-probability futures are handled.

Considering all modes is more conservative and reveals more hidden-risk cases. Filtering by probability reduces conservatism, but may miss risky alternative futures. This directly motivates scenario-based evaluation of safety under multimodal trajectory prediction uncertainty.
