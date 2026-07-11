# GPU Resource Request Note

Checked on 2026-07-05.

## Short Request

Please request access to one CUDA-capable GPU for the cVMD/highD stage of the thesis. The exact memory requirement is not yet verified from a successful local run, so the safest wording is to request a standard research GPU suitable for PyTorch training, with enough time allocation for repeated preprocessing, training, and inference experiments.

## Why GPU Is Needed

The current thesis repository remains CPU-only and does not need GPU for:

- deterministic baseline experiments,
- closed-loop planner comparisons,
- synthetic multimodal uncertainty experiments,
- result plotting and table generation,
- unit tests.

GPU is likely needed or strongly beneficial for:

- VQ-VAE context model training,
- vehicle motion diffusion model training,
- repeated multi-sample diffusion inference,
- cVMDx-style DDIM/multimodal evaluation if the code becomes available.

The cVMD README says the May 2024 implementation currently supports CPU processing and expected GPU support later, while the environment file includes PyTorch and cudatoolkit. The cVMDx paper emphasizes faster DDIM-based inference and practical multi-sample generation, which makes GPU access important for the next thesis stage even if a tiny CPU smoke test works.

## CPU Feasibility

CPU-only work is still useful for:

- checking repository setup,
- inspecting example data,
- understanding output pickle formats,
- testing the adapter/converter on tiny saved outputs,
- keeping the thesis baselines reproducible.

CPU-only full training is not a safe assumption for the MSc timeline.

## Suggested Supervisor / Resource-Team Message

I have a CPU-only thesis evaluation framework working with synthetic trajectory-prediction uncertainty. The next stage is to test cVMD/cVMDx-style diffusion trajectory prediction on highD. The public cVMD code uses PyTorch and documents a two-stage VQ-VAE plus vehicle-motion-diffusion training pipeline. I would like access to one CUDA-capable research GPU for training and repeated multi-sample inference experiments. I do not yet have verified exact memory requirements, so an initial standard GPU allocation plus storage for highD-derived scenario files and model checkpoints would be appropriate.

## Practical Minimum

Before spending GPU time:

1. Obtain highD access.
2. Run a cVMD environment setup check.
3. Run a tiny example-data smoke test.
4. Confirm output files can be converted into this repository's `PredictionSet`.

If these checks fail, continue with the current synthetic multimodal results and document cVMD as planned future integration.
