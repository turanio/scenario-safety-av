# QCNet Real External Artifact Runbook

This runbook prepares the first real external QCNet artifact. It does not make QCNet, Argoverse 2, CUDA, PyTorch Geometric, PyTorch Lightning, or CARLA a dependency of the main thesis environment.

## Goal

Produce one PredictionSet-compatible artifact:

```text
AV2 validation scenario
  -> QCNet released AV2 checkpoint
  -> external QCNet validation/inference
  -> one actor's multimodal prediction
  -> results/qcnet_smoke/qcnet_real_prediction.npz
```

Do not claim a real QCNet thesis result until the artifact is produced from an actual checkpoint and AV2 scenario.

## 1. Confirm External Repo

QCNet is kept outside the project in the sibling `ucl/external_repos` folder:

```bash
python scripts/inspect_qcnet_repo.py --repo-path ../external_repos/QCNet
```

Expected key files:

```text
README.md
environment.yml
train_qcnet.py
val.py
test.py
predictors/
datasets/
```

## 2. Create QCNet Environment

Run this inside the ignored external repo, not in the main `av_safety_eval` environment:

```bash
cd ../external_repos/QCNet
conda env create -f environment.yml
conda activate QCNet
```

If environment creation fails, record the exact error and do not modify `requirements.txt`, `pyproject.toml`, or `environment.yml` in the main thesis repo.

## 3. Place Local Resources

Use ignored local paths:

```text
data/argoverse2/
models/qcnet/
results/qcnet_smoke/
```

Required resources:

- Argoverse 2 Motion Forecasting validation data or a tiny validation subset;
- released QCNet AV2 checkpoint;
- QCNet conda environment;
- GPU if available, though tiny CPU smoke tests may be attempted.

Do not commit AV2 data, checkpoints, processed files, or large pickle outputs.

## 4. Run QCNet Validation or Test

QCNet documents commands similar to:

```bash
python val.py --model QCNet --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
python test.py --model QCNet --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
```

Exact arguments may need adjustment after inspecting `../external_repos/QCNet/README.md`, `val.py`, and `test.py`.

For the first smoke test, prefer:

- one validation scenario or tiny subset;
- batch size 1 if configurable;
- CPU only if GPU is unavailable, with the expectation that it may be slow.

## 5. Export One Prediction Artifact

Use [qcnet_export_prediction_artifact_template.py](/home/turan/ucl/scenario-safety-av/scripts/qcnet_export_prediction_artifact_template.py) as a template. Copy or adapt it inside the QCNet environment after inspecting QCNet output objects.

The final artifact must match [qcnet_prediction_artifact_schema.md](/home/turan/ucl/scenario-safety-av/docs/qcnet_prediction_artifact_schema.md):

```text
scenario_id
target_actor_id
dt
positions        # shape [num_modes, horizon_steps, 2]
probabilities    # shape [num_modes]
coordinate_frame
source
```

Recommended real artifact path:

```text
results/qcnet_smoke/qcnet_real_prediction.npz
```

## 6. Verify the Artifact

Back in the main thesis environment:

```bash
python scripts/verify_qcnet_artifact.py --artifact results/qcnet_smoke/qcnet_real_prediction.npz
```

This writes:

```text
results/qcnet_smoke/qcnet_real_artifact_verification.json
```

## 7. Run Offline Evaluation

```bash
python -m av_safety_eval.experiments.run_qcnet_artifact_evaluation --artifact results/qcnet_smoke/qcnet_real_prediction.npz
```

This is an offline artifact check, not closed-loop CARLA validation.

## 8. Run Top-1 vs Multimodal Artifact Comparison

```bash
python -m av_safety_eval.experiments.run_qcnet_artifact_planner_comparison --artifact results/qcnet_smoke/qcnet_real_prediction.npz
```

This compares top-1 trajectory risk with worst-case multimodal risk using a documented stationary ego rollout.

## 9. Record Outcome

Record:

- QCNet commit hash;
- checkpoint path/name;
- AV2 scenario id;
- target actor id;
- whether the artifact converted successfully;
- offline evaluation result;
- top-1 vs multimodal comparison result;
- any environment or output-extraction errors.

## Current Missing Pieces

The external repo is present locally, but a real artifact still requires:

- QCNet conda environment;
- AV2 validation data or tiny subset;
- released QCNet AV2 checkpoint;
- confirmed output extraction details from QCNet validation/test outputs.
