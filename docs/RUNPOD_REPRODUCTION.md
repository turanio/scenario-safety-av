# RunPod QCNet/AV2 Reproduction

## Scope

This audit records recovery of the pretrained QCNet/Argoverse 2 validation
experiment and the subsequent Stage C probabilistic-risk post-processing. No
model training or fine-tuning was performed.

The analysis remains open-loop center-to-center point-trajectory screening.
QCNet mode weights are not treated as calibrated collision probabilities, and
the probability-weighted distance deficit is not physical collision severity.

## RunPod environment

- Run date: 2026-08-30
- Container: Ubuntu 24.04.3 LTS
- GPU: NVIDIA RTX A5000, 24,564 MiB
- NVIDIA driver: 570.195.03
- Driver-reported CUDA capability: 12.8
- Container disk: 30 GB overlay; 1% used after cache relocation
- Persistent storage root: `/workspace`
- QCNet environment: `/workspace/envs/qcnet`
- Pip cache: `/workspace/cache/pip`
- Matplotlib cache: `/workspace/cache/matplotlib`

All repositories, the environment, checkpoint, AV2 data, processed data,
artifacts, caches, logs, and result outputs were kept under `/workspace`.

## QCNet environment

| Package | Version |
|---|---|
| Python | 3.8.16 |
| PyTorch | 2.0.1+cu118 |
| PyTorch CUDA runtime | 11.8 |
| PyTorch Geometric | 2.3.0 |
| torch-scatter | 2.1.1+pt20cu118 |
| torch-sparse | 0.6.17+pt20cu118 |
| torch-cluster | 1.6.1+pt20cu118 |
| PyTorch Lightning | 2.0.4 |
| torchmetrics | 0.11.4 |
| NumPy | 1.24.3 |
| pandas | 1.4.2 |
| pyarrow | 12.0.1 |
| SciPy | 1.10.1 |
| scikit-learn | 1.2.2 |
| av2 | 0.2.1 |

CUDA verification returned `True`, and PyTorch identified the device as
`NVIDIA RTX A5000` before inference began.

The environment was created and populated with commands equivalent to:

```bash
/workspace/miniforge3/bin/conda create -y \
  -p /workspace/envs/qcnet python=3.8.16 pip=23.1.2

PIP_CACHE_DIR=/workspace/cache/pip /workspace/envs/qcnet/bin/pip install \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118

PIP_CACHE_DIR=/workspace/cache/pip /workspace/envs/qcnet/bin/pip install \
  torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

The remaining versions in the table were installed from the official QCNet
environment specification or pinned PyPI packages. The NVIDIA host driver was
not modified.

## Source and checkpoint

- Thesis repository on RunPod: `/workspace/scenario-safety-av`
- Thesis starting commit: `46c05bc34ada` (`Add CARLA scenario-suite validation results`)
- Official QCNet repository: `/workspace/external_repos/QCNet`
- QCNet commit: `55cacb418cbbce3753119c1f157360e66993d0d0`
- Checkpoint: `/workspace/models/qcnet/QCNet_AV2.ckpt`
- Checkpoint size: 93,083,579 bytes
- Checkpoint SHA-256:
  `b9f852ec888d6fb966a38e3e307af231145b8e1e69cf7b0b9dddd1ae33120f21`

The recovered QCNet working tree includes the pre-existing thesis export and
small-subset compatibility patches. The batch exporter received one additional
narrow change for this recovery: explicit `--device cuda` inference and
device-aware trajectory rotation. Model outputs, target selection, coordinate
transformation, mode probabilities, and artifact schema were not changed.

## AV2 validation subset

- Split: Argoverse 2 Motion Forecasting validation
- Selection: validation scenario directory names sorted lexicographically, then
  the first 500 selected
- Manifest:
  `results/qcnet_server_500/selected_scenario_ids.txt`
- RunPod raw subset: 500 directories, 1,000 files (one scenario parquet and one
  static-map JSON per scenario), approximately 614 MB
- Processed scenarios: 500
- Exported artifacts: 500
- Export skips: 0
- Export retries: 0

The exported artifact-ID set was checked against both the saved manifest and
the historical 500-scenario ranking set; all four sets were identical.

## Commands

The primary RunPod commands were:

```bash
cd /workspace/external_repos/QCNet
QCNET_TINY_DATASET=1 \
PIP_CACHE_DIR=/workspace/cache/pip \
MPLCONFIGDIR=/workspace/cache/matplotlib \
/workspace/envs/qcnet/bin/python -u export_qcnet_batch_scenario_artifacts.py \
  --root /workspace/data/argoverse2 \
  --ckpt_path /workspace/models/qcnet/QCNet_AV2.ckpt \
  --output_dir /workspace/scenario-safety-av/results/qcnet_server_500/artifacts \
  --max_scenarios 500 \
  --device cuda
```

```bash
cd /workspace/scenario-safety-av
PYTHONPATH=src /workspace/envs/qcnet/bin/python \
  src/av_safety_eval/experiments/evaluate_qcnet_batch_artifacts.py \
  --artifact-dir results/qcnet_server_500/artifacts \
  --output-csv results/qcnet_server_500/reproduction_check/qcnet_server_500_ranking.csv \
  --output-json results/qcnet_server_500/reproduction_check/qcnet_server_500_ranking.json

PYTHONPATH=src /workspace/envs/qcnet/bin/python \
  src/av_safety_eval/experiments/run_qcnet_probability_threshold_sweep.py \
  --artifact-dir results/qcnet_server_500/artifacts \
  --per-scenario-csv results/qcnet_server_500/reproduction_check/qcnet_server_500_probability_threshold_sweep.csv \
  --summary-csv results/qcnet_server_500/reproduction_check/qcnet_server_500_probability_threshold_summary.csv \
  --safety-threshold-m 3.0

PYTHONPATH=src MPLCONFIGDIR=/workspace/cache/matplotlib \
/workspace/envs/qcnet/bin/python \
  src/av_safety_eval/experiments/analyze_qcnet_probabilistic_risk.py \
  --artifact-dir results/qcnet_server_500/artifacts \
  --selected-scenario-ids results/qcnet_server_500/selected_scenario_ids.txt \
  --output-dir results/qcnet_server_500/probabilistic_risk \
  --safety-threshold-m 3.0
```

## Mandatory reproduction result

| Check | Reproduced | Required |
|---|---:|---:|
| Scenarios | 500 | 500 |
| Worst-case threshold events | 31 | 31 |
| Top-1 threshold events | 13 | 13 |
| Recorded-ground-truth threshold events | 8 | 8 |
| Hidden-risk cases | 18 | 18 |

The probability-threshold event counts, hidden-risk counts, missed-worst-case
counts, mean eligible-mode counts, and fallback counts reproduced exactly:

| Threshold | Brake | Hidden | Missed worst | Mean modes | Fallback |
|---:|---:|---:|---:|---:|---:|
| 0.000 | 31 | 18 | 0 | 6.000 | 0 |
| 0.001 | 29 | 16 | 2 | 5.462 | 0 |
| 0.010 | 24 | 11 | 7 | 4.950 | 0 |
| 0.030 | 20 | 7 | 11 | 4.508 | 0 |
| 0.050 | 19 | 6 | 12 | 4.138 | 0 |
| 0.100 | 14 | 1 | 17 | 3.424 | 0 |
| 0.200 | 14 | 1 | 17 | 2.246 | 0 |
| 0.300 | 13 | 0 | 18 | 1.256 | 110 |
| 0.500 | 13 | 0 | 18 | 1.000 | 392 |

GPU inference produced small expected floating-point differences from the
historical run. Across top-1, worst-case, ground-truth distance and top-1
probability columns, the largest absolute difference was 0.000358 m in the
worst-case distance for scenario `04f0a78b-fe7d-425f-9ccd-a476900d201f`.
No threshold-event classification changed.

## Stage C probabilistic-risk results

For mode `k`, Stage C computes the minimum ego-to-predicted-target distance
`d_k` over jointly valid ego and focal-target future timesteps. With
`d_safe = 3.0 m`:

- Unsafe probability mass: `sum_k p_k I(d_k < d_safe)`
- Probability-weighted distance deficit:
  `sum_k p_k max(0, d_safe - d_k)`

Summary values:

- Risk-positive scenarios: 31 / 500
- Total unsafe probability mass: 12.412498
- Mean unsafe probability mass over all scenarios: 0.024825
- Mean over risk-positive scenarios: 0.400403
- Median over risk-positive scenarios: 0.135704
- Total probability-weighted distance deficit: 10.222304 m
- Mean severity proxy over all scenarios: 0.020445 m
- Mean over risk-positive scenarios: 0.329752 m
- Median over risk-positive scenarios: 0.070920 m

Threshold retention results are stored in
`probabilistic_risk_threshold_summary.csv`. The percentages retained at the
nine thresholds are:

| Threshold | Unsafe mass retained | Severity retained |
|---:|---:|---:|
| 0.000 | 100.00% | 100.00% |
| 0.001 | 99.97% | 99.97% |
| 0.010 | 99.51% | 99.38% |
| 0.030 | 97.84% | 97.48% |
| 0.050 | 95.87% | 95.85% |
| 0.100 | 88.48% | 85.53% |
| 0.200 | 76.66% | 80.71% |
| 0.300 | 50.43% | 62.83% |
| 0.500 | 39.81% | 36.93% |

## Outputs and logs

Stage C outputs:

- `results/qcnet_server_500/probabilistic_risk/probabilistic_risk_per_scenario.csv`
- `results/qcnet_server_500/probabilistic_risk/probabilistic_risk_threshold_summary.csv`
- `results/qcnet_server_500/probabilistic_risk/probabilistic_risk_summary.md`
- `results/qcnet_server_500/probabilistic_risk/probabilistic_risk_scatter.png`
- `results/qcnet_server_500/probabilistic_risk/probabilistic_risk_threshold_retention.png`

Reproduction outputs:

- `results/qcnet_server_500/reproduction_check/`

Run logs:

- `/workspace/logs/qcnet_export_500.log`
- `/workspace/logs/qcnet_reproduction_evaluation.log`
- `/workspace/logs/qcnet_reproduction_threshold_sweep.log`
- `/workspace/logs/qcnet_reproduction_sanity_check.log`
- `/workspace/logs/qcnet_stage_c_probabilistic_risk.log`

The RunPod pod was left running. The full AV2 validation split was not copied
back from RunPod; only the selected manifest, recreated artifacts, outputs,
logs, source changes, and this audit document are part of the recovery backup.
