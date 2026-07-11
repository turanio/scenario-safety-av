# QCNet External Smoke-Test Path

This document describes the optional QCNet external smoke-test path for future real-model integration.

It does not make QCNet a dependency of the main thesis repository. It does not claim that Argoverse 2 has been evaluated, CARLA has been evaluated, or a real-model thesis result has been completed.

## Goal

Target milestone:

```text
one Argoverse 2 validation scenario
  -> released QCNet checkpoint
  -> 6 multimodal future trajectories + mode probabilities
  -> PredictionSet-compatible artifact
  -> offline artifact evaluation in av_safety_eval
```

## Local Layout

Keep external code, data, and checkpoints out of the main package:

```text
../external_repos/QCNet/
data/argoverse2/
models/qcnet/
results/qcnet_smoke/
```

Use [qcnet_smoke_test.example.yaml](/home/turan/ucl/scenario-safety-av/configs/qcnet_smoke_test.example.yaml) as the local path template. Do not commit a real config with user-specific absolute paths.

## External Setup

From this repository, clone QCNet into the sibling external folder:

```bash
mkdir -p ../external_repos
git clone https://github.com/ZikangZhou/QCNet.git ../external_repos/QCNet
```

Create the QCNet conda environment from the external repository:

```bash
cd ../external_repos/QCNet
conda env create -f environment.yml
conda activate QCNet
```

Download or copy a tiny Argoverse 2 Motion Forecasting validation subset under `data/argoverse2/`, and place the released QCNet AV2 checkpoint under `models/qcnet/`.

Exact QCNet command details may need adjustment after inspecting the external checkout and local AV2 folder structure.

## Inspect the External Repo

From this repository:

```bash
python scripts/inspect_qcnet_repo.py --repo-path ../external_repos/QCNet
```

The command does not clone or import QCNet. It writes:

```text
results/qcnet_smoke/qcnet_repo_inspection.json
```

If QCNet has not been cloned yet, that is an acceptable smoke-test status.

## External QCNet Inference

QCNet documents validation and test commands like:

```bash
python val.py --model QCNet --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
python test.py --model QCNet --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
```

For this thesis smoke test, the important output is not the leaderboard submission. The useful artifact is one scenario's multimodal future positions and mode probabilities saved in the simple `.npz` format described in [qcnet_prediction_artifact_schema.md](/home/turan/ucl/scenario-safety-av/docs/qcnet_prediction_artifact_schema.md).

## Converter Test Without QCNet

Before real QCNet inference works, create a fake artifact for converter testing only:

```bash
python scripts/create_fake_qcnet_artifact.py
```

This writes:

```text
results/qcnet_smoke/fake_qcnet_prediction.npz
```

The fake artifact is not thesis evidence and must not be described as QCNet integrated.

## Offline Artifact Evaluation

Run:

```bash
python -m av_safety_eval.experiments.run_qcnet_artifact_evaluation --artifact results/qcnet_smoke/fake_qcnet_prediction.npz
```

The current evaluation uses a documented stationary ego rollout at the origin and reports mode count, horizon, probability sum, and worst-case minimum distance. It is only a sanity check for the artifact path.

## Success

The external smoke path is successful when:

- QCNet repo inspection succeeds;
- one real AV2/QCNet artifact is saved as `.npz`;
- `load_qcnet_npz_prediction()` converts it into `PredictionSet`;
- offline artifact evaluation runs;
- no QCNet, AV2, CUDA, or CARLA dependency is added to the main environment.

## Failure Modes

Expected non-blocking failures:

- QCNet repository missing locally;
- AV2 validation data not downloaded;
- checkpoint not present;
- CPU smoke test too slow.

These should be reported honestly. The current synthetic and thesis-ready results remain the working baseline.

## Next Step: Real External QCNet Artifact

The real external QCNet artifact preparation is documented in [qcnet_real_external_runbook.md](/home/turan/ucl/scenario-safety-av/docs/qcnet_real_external_runbook.md).

Use these helper files:

- [qcnet_export_prediction_artifact_template.py](/home/turan/ucl/scenario-safety-av/scripts/qcnet_export_prediction_artifact_template.py)
- [verify_qcnet_artifact.py](/home/turan/ucl/scenario-safety-av/scripts/verify_qcnet_artifact.py)

The export template is intended to be adapted after inspecting QCNet's validation/test output objects. The verification script works with fake and real PredictionSet-compatible artifacts.
