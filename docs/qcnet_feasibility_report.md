# QCNet Feasibility Report

Checked on 2026-07-05.

## Decision

QCNet is the most practical real-model trajectory prediction path for this thesis stage. It is not a diffusion model and should be described as QCNet multimodal prediction. The thesis contribution remains the scenario-based safety evaluation framework and the SafeIO-style uncertainty-aware planning layer, not QCNet itself.

QCNet should not become a hard dependency of the main `av_safety_eval` environment. It should be run in a separate external checkout and optional conda environment, then connected to this repository through a small adapter/converter once a smoke test works.

## Sources Checked

- QCNet official repository: https://github.com/ZikangZhou/QCNet
- QCNet environment file: https://raw.githubusercontent.com/ZikangZhou/QCNet/main/environment.yml
- QCNet paper page linked from the repository: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.pdf
- Argoverse 2 user guide: https://argoverse.github.io/user-guide/
- Argoverse 2 dataset page: https://www.argoverse.org/av2.html
- Argoverse 2 API repository: https://github.com/argoverse/av2-api
- CARLA documentation: https://carla.readthedocs.io/en/latest/

## Repository Availability

The official QCNet repository is reachable at `ZikangZhou/QCNet`. It is public, licensed under Apache-2.0, and describes itself as the official implementation of the CVPR 2023 paper `Query-Centric Trajectory Prediction`.

The repository directly supports Argoverse 2. Its README reports rank-1 results on Argoverse 1 single-agent, Argoverse 2 single-agent, and Argoverse 2 multi-agent motion forecasting benchmarks. It contains `train_qcnet.py`, `val.py`, `test.py`, Argoverse 2 dataset code, PyTorch Lightning model code, and a checkpoint link for Argoverse 2 marginal prediction.

## Environment

The published QCNet environment is separate from the current thesis package environment. It uses Python 3.8.16, PyTorch 2.0.1 with CUDA 11.8, PyTorch Geometric, PyTorch Lightning 2.0.x, TorchMetrics, AV2 API usage, and other ML tooling.

The current thesis package intentionally remains Python 3.11+, CPU-only, and lightweight. Therefore QCNet should be isolated under an external repo path such as:

```text
../external_repos/QCNet/
```

## Checkpoints

The QCNet README lists a released `QCNet_AV2` checkpoint for Argoverse 2 marginal prediction. This is a major practical advantage over the current cVMD path, where no clearly reusable pretrained vehicle-motion-diffusion checkpoint was found during the previous feasibility check.

The checkpoint still needs to be downloaded manually and kept out of git, for example under ignored `models/` or outside the repository.

## Training, Validation, and Test Commands

The README documents:

```bash
python train_qcnet.py --root /path/to/dataset_root/ --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --devices 8 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150
python val.py --model QCNet --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
python test.py --model QCNet --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt
```

The README says training consumes about 160 GB GPU memory in the cited 8 x RTX 3090 setup and that first-run preprocessing can take several hours. This makes training unrealistic as the first MSc step. Validation or test with the released checkpoint is the preferred smoke test.

## Output Shape and Conversion Feasibility

QCNet produces multimodal trajectories. Its model has `num_modes`, defaulting to 6, and `num_future_steps`, commonly 60 for Argoverse 2. The test code converts refined local predictions back into global coordinates and applies softmax to mode logits `pi`, producing per-mode probabilities for each evaluated actor.

This can be converted into this repository's `PredictionSet`:

```text
QCNet global trajectories: num_modes x horizon_steps x 2
QCNet mode scores: num_modes
av_safety_eval: list[Trajectory] + probabilities
```

The conversion is feasible. The risky part is not the output shape; it is setting up the QCNet/AV2 environment, choosing the target actor, and matching coordinate frames for planning.

## GPU and CPU Feasibility

Training requires GPU resources. The repository explicitly describes a large multi-GPU training setup.

Inference/validation may be possible on CPU for a tiny subset if batch size and workers are reduced, but this should be treated as a smoke test only. Practical repeated inference on AV2 or any online-style loop should use GPU.

## Comparison With cVMD

QCNet is more practical than cVMD for the main thesis implementation path because:

- official public implementation is reachable;
- Argoverse 2 support is explicit;
- AV2 checkpoint availability is documented;
- output is already multimodal with mode scores;
- the dataset is public and directly downloadable, though large;
- the project avoids making diffusion training the critical path.

cVMD/highD remains useful as a secondary diffusion exploration track, but QCNet is the better path for a defensible real-model result within the thesis timeline.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Full AV2 dataset is large | Medium/High | Use a small validation subset for smoke tests; keep data out of git |
| QCNet environment is heavy | Medium | Use external conda environment; do not add to main requirements |
| Training is too expensive | High | Use released AV2 checkpoint; do not train from scratch first |
| CPU inference is too slow | Medium | Use CPU only for smoke tests; request GPU for repeated experiments |
| Coordinate conversion mistakes | High | Validate with one AV2 scenario visualization before planner use |
| Direct QCNet-CARLA integration is risky | Medium/High | Start offline with AV2, then recreate scenario patterns in CARLA |

## Recommended Next Step

Proceed with a QCNet smoke test using the released Argoverse 2 checkpoint and a tiny AV2 validation subset. The first milestone should produce one saved `PredictionSet`-compatible artifact for one scenario, not a closed-loop CARLA demo.
