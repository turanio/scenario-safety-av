# Selected Hidden-Risk Scenario Review

## Selection Basis

The 500-scenario QCNet ranking contains 18 hidden-risk cases under the 3.0 m point-distance threshold. A hidden-risk case has `top1_action == NO_BRAKE` and `worst_case_action == BRAKE`.

Cases whose worst-case minimum occurs at the final horizon step were excluded because at least three non-final alternatives were available. The three selections represent: the smallest non-final minimum, the smallest non-final minimum with triggering-mode p >= 0.01, and the highest triggering-mode probability among the remaining non-final cases. AV2 map context was available for all three and was visually reviewed.

## Selected Cases

| Scenario | Role | Top-1 min (m) | Worst-case min (m) | Ground truth min (m) | Worst p | Worst step | Use |
|---|---|---:|---:|---:|---:|---:|---|
| `001749f1-bc1c-47fb-a13f-9ab1f2c050a8` | balanced_severity | 3.317 | 0.448 | 3.406 | 0.024 | 51 | primary_case_study |
| `00e2cd17-25bc-42f2-8f33-17ae24d17a5f` | higher_probability_threshold_case | 3.038 | 2.868 | 2.941 | 0.141 | 19 | secondary_case |
| `032618a4-3f4b-456a-b575-17297fcc1ceb` | minimum_distance_tail_case | 5.503 | 0.272 | 5.323 | 7.16e-06 | 54 | appendix_only |

## Interpretation

### `001749f1-bc1c-47fb-a13f-9ab1f2c050a8`

Smallest non-final worst-case minimum among candidates whose triggering mode has p >= 0.01. Top-1 remains at 3.317 m while mode 5 (p=0.024) reaches 0.448 m at step 51. The map view shows the interaction around an intersection and makes the separation between the recorded ego path and target alternatives readable.

[Map and actor context](map_actor_context_hidden_risk_001749.png) | [Closest-interaction zoom](closest_interaction_hidden_risk_001749.png) | [Distance over time](distance_over_time_hidden_risk_001749.png)

### `00e2cd17-25bc-42f2-8f33-17ae24d17a5f`

Highest worst-case-mode probability among the remaining non-final hidden-risk candidates. Top-1 remains at 3.038 m while mode 3 (p=0.141) reaches 2.868 m at step 19. The map view shows a same-corridor close interaction, with all three minima at the same early timestep.

[Map and actor context](map_actor_context_hidden_risk_00e2cd.png) | [Closest-interaction zoom](closest_interaction_hidden_risk_00e2cd.png) | [Distance over time](distance_over_time_hidden_risk_00e2cd.png)

### `032618a4-3f4b-456a-b575-17297fcc1ceb`

Smallest non-final worst-case point distance; retained to show extreme-tail sensitivity. Top-1 remains at 5.503 m while mode 5 (p=7.16e-06) reaches 0.272 m at step 54. The map view shows a lane-following interaction, but the triggering mode has extremely low probability and occurs late in the horizon.

[Map and actor context](map_actor_context_hidden_risk_032618.png) | [Closest-interaction zoom](closest_interaction_hidden_risk_032618.png) | [Distance over time](distance_over_time_hidden_risk_032618.png)

## Recommendation

`001749f1-bc1c-47fb-a13f-9ab1f2c050a8` is the strongest headline hidden-risk example. It combines a sub-metre worst-case point distance, a non-final minimum, a triggering probability above the selection floor, and readable intersection context. Its ground-truth path remains above the threshold, so the case demonstrates multimodal forecast sensitivity rather than a recorded near miss.

`00e2cd17-25bc-42f2-8f33-17ae24d17a5f` is a useful secondary case because the worst-case mode has the highest probability in the selected set and ground truth also falls just below the threshold. It is borderline: all three minima lie close to 3.0 m, so its classification is sensitive to the chosen point-distance threshold.

`032618a4-3f4b-456a-b575-17297fcc1ceb` should be appendix-only. Its very small worst-case distance is useful for showing worst-case-filter sensitivity, but the triggering mode has extremely low probability and occurs late in the forecast horizon.

## Scope And Caveats

These figures are open-loop point-trajectory analyses using QCNet multimodal predictions and the recorded AV2 ego future. They do not simulate a controller response and are not closed-loop safety proof.

The distances are between trajectory points, not oriented vehicle footprints. A value below 1 m or near zero is not evidence of a confirmed collision. Map context supports qualitative interpretation but does not replace actor-envelope or vehicle-geometry checking.

The selected cases are illustrative examples from this 500-scenario sample. They motivate uncertainty-aware evaluation but do not prove that a forecasting model or safety filter improves closed-loop safety.
