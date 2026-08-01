# QCNet AV2 500-Scenario Server Evaluation Summary

## Setup

- Server: edale
- Predictor: released QCNet AV2 checkpoint
- Dataset: Argoverse 2 validation split
- Sample size: 500 scenarios
- Prediction modes: 6
- Horizon: 60 future steps
- Timestep: 0.1 s
- Safety threshold: 3.0 m near-miss threshold
- Evaluation type: open-loop distance-based evaluation

## Main Results

| Metric | Count |
|---|---:|
| Total evaluated scenarios | 500 |
| Worst-case multimodal near-miss cases | 31 |
| Top-1 near-miss cases | 13 |
| Ground-truth near-miss cases | 8 |
| Hidden-risk cases | 18 |

A hidden-risk case means the top-1 predicted mode remains above the 3.0 m threshold, but at least one alternative predicted mode falls below the threshold.

## Interpretation

The 500-scenario evaluation shows that top-1 prediction alone misses a meaningful number of risky multimodal futures. In this sample, 18 scenarios were hidden-risk cases. This supports the thesis motivation that safety evaluation should consider multimodal trajectory prediction uncertainty, not only the most likely future.

The results are still open-loop and distance-based. They should not be interpreted as closed-loop safety proof or confirmed physical collisions.

## Representative Cases

| Scenario | Type | Top-1 min distance | Worst-case min distance | Ground-truth min distance | Interpretation |
|---|---:|---:|---:|---:|---|
| 001749f1-bc1c-47fb-a13f-9ab1f2c050a8 | Hidden risk | 3.317 m | 0.448 m | 3.406 m | Top-1 appears safe, but a lower-probability mode creates a severe predicted close interaction |
| 0091bad9-e7b2-4c07-aa12-6b5fd03c63d2 | High-confidence close interaction | 3.181 m | 3.136 m | 3.228 m | Top-1, worst-case, and ground truth are closely aligned above the threshold |
| 00351569-255c-433e-b97b-e2a844d1b6e0 | Real near-miss | 2.597 m | 2.045 m | 2.162 m | Prediction and ground truth both fall below the near-miss threshold |

## SafeIO-Style Filter Result

The lightweight SafeIO-style filter comparison shows three different behaviours:

1. In the hidden-risk case, worst-case filtering brakes, but top-1 and probability-aware filtering do not.
2. In the high-confidence close interaction case, all filters return no brake.
3. In the real near-miss case, all filters brake.

This demonstrates the trade-off between worst-case conservatism and probability-aware filtering under multimodal prediction uncertainty.
