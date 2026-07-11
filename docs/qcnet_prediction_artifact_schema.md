# QCNet Prediction Artifact Schema

This schema defines the small dependency-free artifact expected by `av_safety_eval` after external QCNet inference.

The artifact is not a QCNet checkpoint, not AV2 data, and not thesis evidence by itself. It is only the bridge from an external QCNet run into this repository's `PredictionSet` interface.

## Preferred Format

Use NumPy `.npz` because the main payload is array data.

Required keys:

```text
scenario_id
target_actor_id
dt
positions
probabilities
coordinate_frame
source
```

## Key Definitions

`scenario_id`: scalar string identifying the AV2 scenario or smoke-test scenario.

`target_actor_id`: scalar string identifying the actor predicted by QCNet.

`dt`: positive scalar float in seconds. Argoverse 2 motion forecasting is sampled at 10 Hz, so the expected value is `0.1`.

`positions`: float array with shape `[num_modes, horizon_steps, 2]`. Units must be meters. The last dimension is `[x, y]`.

`probabilities`: float array with shape `[num_modes]`. Values must be non-negative and sum to 1 within a small numerical tolerance.

`coordinate_frame`: scalar string naming the coordinate frame. The first integration should use global AV2 coordinates if QCNet already exports them.

`source`: scalar string describing where the artifact came from, for example `qcnet_av2_validation_smoke_test`.

## Rules

- Horizon order must be future time order.
- Do not silently mix local agent-centric coordinates and global map coordinates.
- Do not store raw AV2 data or QCNet checkpoints in this artifact.
- Do not commit real artifacts if they are large or contain dataset-derived content.
- Fake artifacts must be labelled as converter tests only.

## Converter

Use:

```python
from av_safety_eval.predictors.qcnet_output_converter import load_qcnet_npz_prediction

prediction = load_qcnet_npz_prediction("results/qcnet_smoke/fake_qcnet_prediction.npz")
```

The converter returns a `PredictionSet` containing one `Trajectory` per mode and the corresponding probabilities.
