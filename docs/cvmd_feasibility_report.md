# cVMD / cVMDx Feasibility Report

Checked on 2026-07-05.

## Decision

cVMD is a feasible candidate for the thesis diffusion-prediction stage, but it should be integrated only after highD access and a separate external cVMD environment are available. cVMDx is attractive for the thesis story because it targets faster multimodal inference, but the public cVMD repository's cVMDx link currently appears unavailable, so cVMDx cannot be treated as an implementation dependency yet.

The current repository should keep the synthetic multimodal predictor as the minimum viable result while cVMD/highD feasibility is resolved.

## Sources Checked

- cVMD GitHub repository: https://github.com/MB-Team-THI/conditioned-vehicle-motion-diffusion
- cVMD environment file: https://raw.githubusercontent.com/MB-Team-THI/conditioned-vehicle-motion-diffusion/main/environment.yml
- cVMD paper: https://arxiv.org/abs/2405.14384
- cVMDx paper: https://arxiv.org/abs/2602.21319
- highD dataset page: https://levelxdata.com/highd-dataset/
- highD format description: https://levelxdata.com/wp-content/uploads/2023/10/highD-Format.pdf

## Repository Availability

The repository `MB-Team-THI/conditioned-vehicle-motion-diffusion` is reachable and public. It has an MIT license, a README, an `environment.yml`, example data, `vqvae`, `vmduc`, and example result/checkpoint-style folders. The GitHub page showed 4 commits, no published releases, and one open issue at the time of the check.

The README describes the project as the official PyTorch implementation of cVMD. It states that the May 2024 release currently supports CPU processing and that GPU processing was expected later. The `environment.yml` includes PyTorch 1.7.0, torchvision 0.8.1, cudatoolkit 11.8, Python 3.8.5, and older scientific Python packages. This is not compatible with the main thesis package environment, which currently targets Python 3.11+ and CPU-only dependencies.

## cVMDx Availability

The README points to an improved `cvmdx` implementation and says it supports GPU acceleration, DDIM sampling, and multimodal trajectory-hypothesis generation. However, the linked GitHub target currently resolves as unavailable during this check, and the repository has an open issue titled `cvmdx link is broken (GPU/DDIM/multimodal)`.

The cVMDx paper is available on arXiv and describes DDIM sampling, faster inference, and multimodal prediction on highD. That supports cVMDx as a research reference, but not yet as a runnable integration target.

## Training and Testing Scripts

The cVMD README documents a two-stage workflow:

1. Train the VQ-VAE context-conditioning model from `vqvae`.
2. Run VQ-VAE inference/MLE to produce embedding and codebook metadata.
3. Train the vehicle motion diffusion module from `vmduc`.
4. Test the trained model with `vmduc/main_test.py`.

The README describes output artifacts including VQ-VAE checkpoints, embedding metadata, VMD checkpoints, and test results stored as pickle files. This suggests inference is possible after training artifacts exist, but the public README does not present a simple plug-and-play inference API for an external planner loop.

## Pretrained Weights

No clearly reusable pretrained VMD model weights were found during this repository-page inspection. The repository includes example folders such as `vqresults/example/epoch=000052` and `vmduc/ckpts/example/results`, but the README test command still expects a user-supplied VMD checkpoint path such as `vmduc/ckpts/run_name_vmd/model=yyy.pt`.

Practical implication: assume training or obtaining weights from the authors is required before cVMD can replace the synthetic multimodal predictor.

## Dataset Dependency

cVMD was tested on highD and expects preprocessed `.mat` scenario files under dataset folders such as `data/highD/train/class0`, `class1`, and `class2`. The repository README says the included data examples are computer-generated for illustration and are not taken from highD.

The highD dataset itself requires manual access through the dataset website. It is free for academic and research purposes after approval, but redistribution is restricted. No highD files should be committed to this repository.

## Feasibility Answer

Can we realistically use cVMD/cVMDx + highD as the proposed diffusion-based trajectory prediction component?

Yes, with caveats:

- cVMD is reachable, documented, and aligned with highD highway lane-change prediction.
- cVMD should be run outside the main thesis environment because it needs Python 3.8-era dependencies and PyTorch.
- Inference-only integration is possible only after valid checkpoints and preprocessed highD-style files exist.
- cVMDx is currently a paper-level and README-referenced target, but public code availability is unresolved.
- GPU access should be requested before committing thesis time to full training or multi-sample inference experiments.

## Risk Table

| Risk | Impact | Mitigation |
|---|---|---|
| No pretrained weights | High | Train a small model if feasible, ask authors/supervisor about weights, or keep synthetic multimodal predictor as fallback |
| highD access delay | Medium/High | Request highD access early; continue using synthetic scenarios meanwhile |
| cVMDx code unavailable | Medium | Use cVMD repo as first runnable target or choose a fallback diffusion model |
| GPU unavailable | High | Run only baseline/synthetic locally; request GPU for training and multi-sample inference |
| Output format hard to adapt | Medium | Build a conversion wrapper from cVMD result pickles to `PredictionSet` |
| Environment mismatch | Medium | Keep cVMD in a separate conda environment or external repo checkout |
| Timeline risk | High | Preserve current synthetic results as the minimum viable thesis contribution |

## Recommended Next Step

Request highD access and GPU resources now. In parallel, clone cVMD into an ignored external folder, run the optional local inspection script, and attempt a CPU smoke test with the repository's example data in the cVMD conda environment. Do not integrate it into `av_safety_eval` until an inference command can produce target trajectories that can be converted to `num_modes x horizon_steps x 2`.
