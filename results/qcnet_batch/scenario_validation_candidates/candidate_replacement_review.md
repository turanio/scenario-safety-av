# QCNet Candidate Replacement Review

## Scope

This review compares three possible replacements for scenario `0058ed53-93bf-42a7-9bba-6df3f6ce20f5` in the selected QCNet examples. The comparison uses the recorded AV2 ego trajectory, the focal actor's ground-truth future, six QCNet point-trajectory modes, and the local AV2 map. Timestep indices are zero-based; the first predicted point is at 0.1 s.

These are open-loop point-trajectory analyses, not closed-loop safety results. The recorded ego trajectory does not respond to a predicted risk. Distances are between trajectory points and do not include actor envelopes or vehicle footprints, so none of the cases establishes a collision.

## Comparison

| Scenario | Candidate type | Top-1 / worst / ground truth min (m) | Minimum steps (top-1 / worst / GT) | Top-1 probability | Worst-mode probability | Horizon-end worst minimum | Recommended use |
|---|---|---:|---:|---:|---:|---|---|
| `005f3e26-75cb-43a5-b61a-89343d490fdf` | Borderline near-miss threshold | 3.036 / 3.003 / 3.606 | 27 / 24 / 38 | 0.225 | 0.107 | No | `secondary_case` |
| `0091bad9-e7b2-4c07-aa12-6b5fd03c63d2` | High-confidence close interaction | 3.181 / 3.136 / 3.228 | 16 / 19 / 18 | 0.912 | 0.000377 | No | `primary_case_study` |
| `00170a57-eb8a-4947-8b7d-7b12af7af2db` | Moderate close interaction | 3.427 / 3.261 / 3.362 | 39 / 49 / 43 | 0.532 | 0.000788 | No | `appendix_only` |

## Candidate Assessment

### 005f3e: Borderline Threshold Case

The top-1 and worst-case predictions both approach the 3.0 m threshold before the middle of the horizon, while ground truth remains farther away at 3.606 m. The mapped trajectories show a plausible close interaction in the road corridor rather than an isolated endpoint convergence. This is useful for discussing sensitivity to a chosen threshold, but the lower top-1 probability and larger ground-truth separation make it weaker as the main replacement.

### 0091bad: High-Confidence Close Interaction

This candidate has the strongest agreement across evidence sources. Top-1, worst-case, and ground-truth minimum distances occupy a narrow 0.092 m range, and their minima occur within three timesteps around 1.7-2.0 s. The map view places the trajectories coherently within the same road corridor. The top-1 probability is 0.912, so the close interaction is represented by QCNet's dominant mode as well as ground truth. The nominal worst-case mode has very low probability, but the case does not depend on that mode because top-1 and ground truth report similar separation.

### 00170a: Moderate Close Interaction

The three minimum distances are reasonably consistent and all occur before the final timestep. The mapped interaction is plausible, but the minimum separation stays above 3.0 m and occurs later than in `0091bad`. The worst-case mode also has negligible probability. The scenario is suitable as supporting or appendix evidence, but it adds less to the headline comparison.

## Recommendation

Use `0091bad9-e7b2-4c07-aa12-6b5fd03c63d2` as the replacement for `0058ed53` in the three headline selected examples. It better satisfies the stated criteria: the closest interaction occurs well before the horizon end, top-1 and ground truth are closely aligned with the worst-case distance, the dominant mode has high probability, and the map context supports a coherent road interaction.

Keep `0058ed53` in the analysis as an appendix or sensitivity example of a large multimodal gap whose minimum occurs at the final horizon step. Do not delete it or interpret its sub-meter point distance as a collision. Retain `005f3e` as a secondary threshold-sensitivity case and `00170a` as appendix-only supporting evidence.

This recommendation improves the robustness and interpretability of the selected examples; it does not demonstrate that QCNet or a planner improves safety.
