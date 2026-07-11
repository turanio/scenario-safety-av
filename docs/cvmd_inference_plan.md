# cVMD Inference Plan

Checked on 2026-07-05.

## Objective

Replace the current `SyntheticMultimodalPredictor` with a real diffusion predictor only after cVMD can produce trajectory samples that fit this repository's predictor interface:

```text
PredictionSet
  trajectories: list[Trajectory]
  probabilities: list[float] | None

trajectory positions shape: horizon_steps x 2
planner-facing stacked shape: num_modes x horizon_steps x 2
```

## Current Feasibility

Inference-only use is plausible but not immediately ready.

The cVMD README documents `vmduc/main_test.py` with a `--model_path` argument and a `--vqvae_dir` argument. That indicates test/inference can run after a trained VMD checkpoint and VQ-VAE embedding/codebook artifacts exist. The README also says test output is written to a results pickle containing generated predictions and ground truth information.

The public repository inspection did not find clearly documented pretrained VMD weights. Therefore, assume one of these is needed:

1. Train cVMD on highD.
2. Obtain compatible pretrained weights from the authors/supervisor.
3. Use the included computer-generated examples only for a smoke test, not thesis claims.

## Proposed Integration Boundary

The main repository now contains `CVMDAdapter` at:

```text
src/av_safety_eval/predictors/cvmd_adapter.py
```

The adapter is intentionally dependency-free. It accepts optional paths:

```python
CVMDAdapter(model_path="...", config_path="...")
```

It raises `NotImplementedError` until a separate cVMD environment can produce verified outputs.

## External Environment Strategy

Do not install cVMD into the main thesis environment. The main environment currently targets Python 3.11+ and CPU-only testing. cVMD's environment file specifies Python 3.8.5 and older PyTorch-era packages.

Recommended layout:

```text
../external_repos/conditioned-vehicle-motion-diffusion/
data/highD/                 # ignored local data, or outside repo
models/cvmd/                # ignored local checkpoints, or outside repo
```

Use cVMD's own conda environment for training/testing. Use this repository only for adapter conversion, planner evaluation, and metrics once output files are produced.

## Smoke-Test Steps

1. Request and receive highD access.
2. Clone cVMD outside the main package, for example under `../external_repos/`.
3. Create the cVMD conda environment from its `environment.yml`.
4. Run `scripts/inspect_cvmd_repo.py --repo-path <local-cvmd-path>` from this repository to confirm local files.
5. Try the cVMD example data first.
6. Confirm whether `vmduc/main_test.py` can run with included example artifacts.
7. Inspect the generated pickle structure.
8. Write a small converter that maps generated predictions into `Trajectory` objects.
9. Run one closed-loop evaluation with `CVMDAdapter` behind an optional flag.

## Expected Converter Contract

The converter should produce:

- target `agent_id`
- one `Trajectory` per generated sample or clustered mode
- `positions` with columns `[x, y]`
- `dt`, likely `1 / 25` for highD unless resampled
- optional probabilities from sample frequencies or cVMDx/GMM mode weights

If cVMD produces many raw samples without probabilities, the first integration can use equal probabilities and document that limitation.

## Timeline Risk

Training is the high-risk item. Without pretrained weights, cVMD requires highD preprocessing, VQ-VAE training, MLE/embedding generation, VMD training, and test/inference conversion before it can replace the synthetic predictor. This is realistic only if highD access and GPU resources arrive early.

The fallback thesis path remains valid: keep the synthetic multimodal experiments as the controlled proof of uncertainty-aware planning behavior, and present cVMD/highD as the planned realism extension if full training does not fit the MSc timeline.
