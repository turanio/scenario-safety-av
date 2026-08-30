# RunPod QCNet 10,000-Scenario Independent Validation

## Status and scope

The independent 10,000-scenario QCNet/Argoverse 2 (AV2) validation run completed
successfully on 30 August 2026. The cohort was fixed before inference, excludes the
historical 500-scenario development/reproduction cohort, and passed a mandatory
artifact-level integrity gate before aggregate analysis.

This is open-loop point-trajectory safety screening. It evaluates how treating the
six QCNet futures differently changes screening and intervention decisions relative
to the one recorded AV2 future. It is not calibrated collision-probability estimation,
exact vehicle-footprint collision checking, collision avoidance, or closed-loop safety
validation. QCNet was used only for pretrained inference; no training, fine-tuning, or
post-result threshold tuning was performed.

## Reproducibility record

| Item | Value |
|---|---|
| Run date (UTC) | 2026-08-30 |
| RunPod GPU | NVIDIA RTX A5000, 24,564 MiB |
| NVIDIA driver | 570.195.03 |
| Container OS | Ubuntu 24.04.3 LTS |
| Kernel | Linux 6.8.0-52-generic x86_64 |
| Python | 3.8.16 |
| PyTorch | 2.0.1+cu118 |
| PyTorch CUDA runtime | 11.8 (`torch.cuda.is_available() = True`) |
| PyTorch Geometric | 2.3.0 |
| PyTorch Lightning | 2.0.4 |
| torchmetrics | 0.11.4 |
| NumPy / pandas / SciPy | 1.24.3 / 1.4.2 / 1.10.1 |
| pyarrow / scikit-learn / av2 | 12.0.1 / 1.2.2 / 0.2.1 |
| QCNet commit | `55cacb418cbbce3753119c1f157360e66993d0d0` |
| Checkpoint | `/workspace/models/qcnet/QCNet_AV2.ckpt` |
| Checkpoint SHA-256 | `b9f852ec888d6fb966a38e3e307af231145b8e1e69cf7b0b9dddd1ae33120f21` |
| Local thesis commit before run | `c51d068fcde0609902f861e6f3bdbcbe294f9350` |
| RunPod thesis checkout HEAD | `46c05bc34ada9f90031caee5f0a1686077f0c8f1` plus synchronized run files |
| Historical manifest | `results/qcnet_server_500/selected_scenario_ids.txt` |
| 10k manifest | `results/qcnet_server_10000/selected_scenario_ids.txt` |
| 10k manifest SHA-256 | `0cbca1a73627430cf712623d90589ccf6e17bc51ab0eb3ee8a0dc18ff21ae7a0` |

The RunPod checkout was older than the local repository HEAD, so the required source
files were synchronized explicitly. The immutable local starting commit above is the
canonical thesis provenance; the synchronized files are included in the final commit.

## Cohort construction

Available AV2 validation scenario directory names were enumerated, the historical
manifest was removed, and `random.Random(42).sample(...)` selected without replacement
from the sorted remaining candidates. Only after sampling were selected IDs sorted for
deterministic processing order.

| Cohort check | Result |
|---|---:|
| Available AV2 validation IDs | 24,988 |
| Historical manifest IDs | 500 |
| Historical IDs present and excluded | 500 |
| Independent candidates | 24,488 |
| Seed | 42 |
| Selected / unique selected | 10,000 / 10,000 |
| Historical overlap | 0 |
| Raw scenario files transferred | 20,000 |

The manifest was not replaced or supplemented after inference began.

## Storage and runtimes

Before transfer, the 500-run footprint was scaled to estimate about 4.7 GB for raw,
processed, and artifact data, or about 14 GB with a conservative 3x margin. `/workspace`
had about 158 TB free on its network volume, so the run fit safely. Substantial data,
the environment, repository, checkpoint, and caches remained under `/workspace`.

| Stage / footprint | Result |
|---|---:|
| Selected raw AV2 data | 2,542,474,865 bytes (2.37 GiB) |
| Processed AV2 data | 2,034,186,403 bytes (1.89 GiB) |
| Exported artifact directory | 98,065,181 bytes (93.5 MiB) |
| Raw transfer runtime | 532 s (8 min 52 s) |
| Preprocessing runtime | 1,856 s (30 min 56 s) |
| GPU inference/export wrapper runtime | 1,658 s (27 min 38 s) |
| Exporter's measured inference loop | 1,619.44 s |
| Integrity-gate runtime | 277 s (4 min 37 s) |
| Quantitative analysis runtime | 469 s (7 min 49 s) |
| Integrity plus analysis | 746 s (12 min 26 s) |

The initial preprocessing wrapper exited before processing because `/usr/bin/time` was
not installed. The unchanged command was rerun with shell timestamps; it processed all
10,000 scenarios. This was an orchestration failure, not a scenario/data failure.

## Export and integrity gate

| Export counter | Result |
|---|---:|
| Manifest / processed / successful | 10,000 / 10,000 / 10,000 |
| New / resumed artifacts | 10,000 / 0 |
| Skips / retries / failures | 0 / 0 / 0 |
| Missing dataset entries / duplicates | 0 / 0 |
| Device | CUDA |

The downstream analyses were run only after the following gate passed:

| Integrity check | Result |
|---|---:|
| Manifest / unique manifest IDs | 10,000 / 10,000 |
| Artifacts / unique artifact IDs | 10,000 / 10,000 |
| Artifact IDs exactly equal manifest IDs | Pass |
| Historical overlap | 0 |
| Six-mode artifacts | 10,000 |
| 60-step artifacts | 10,000 |
| Artifacts with jointly valid future steps | 10,000 |
| Finite non-negative, approximately normalized probabilities | Pass |
| Failed integrity rows | 0 |
| Overall status | **Pass** |

## Standard 10k results

The frozen event threshold is center distance `< 3.0 m` over jointly valid future
timesteps. The recorded-future result is a reference to the one realized AV2 future,
not complete ground-truth risk.

| Event | Count | Rate |
|---|---:|---:|
| Top-1 threshold event | 225 | 2.25% |
| Worst-case threshold event | 558 | 5.58% |
| Recorded-future threshold event | 159 | 1.59% |
| Hidden risk (worst-case yes, top-1 no) | 333 | 3.33% |

Across all scenarios, the mean multimodal distance gap was 2.337 m and the median was
0.098 m. Among the 333 hidden-risk scenarios, the mean was 5.419 m; P25, P50, P75,
P90, P95, and P99 were 2.136, 4.990, 7.494, 11.011, 12.653, and 19.050 m.

## Frozen probability-filter sweep

`theta=0.05` remains the predefined representative point; it was not selected from
these results. Retained expected risk is the normalized expected distance-deficit proxy.

| theta | Triggered | Rate | Hidden found | Missed worst | Fallback | Mean modes | Mean mass | Unsafe mass retained | Unsafe retained | Expected risk retained | Expected retained |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 558 | 5.58% | 333 | 0 | 0 | 6.000 | 1.000000 | 238.491988 | 100.00% | 43.237945 | 100.00% |
| 0.001 | 515 | 5.15% | 290 | 43 | 0 | 5.560 | 0.999896 | 238.455694 | 99.98% | 43.224132 | 99.97% |
| 0.010 | 451 | 4.51% | 226 | 107 | 0 | 5.052 | 0.997574 | 237.617966 | 99.63% | 42.962308 | 99.36% |
| 0.030 | 398 | 3.98% | 173 | 160 | 0 | 4.550 | 0.988005 | 234.552276 | 98.35% | 42.023412 | 97.19% |
| 0.050 | 374 | 3.74% | 149 | 184 | 0 | 4.194 | 0.973952 | 230.203972 | 96.52% | 40.691789 | 94.11% |
| 0.100 | 330 | 3.30% | 105 | 228 | 0 | 3.470 | 0.920335 | 216.659786 | 90.85% | 38.300250 | 88.58% |
| 0.200 | 282 | 2.82% | 57 | 276 | 18 | 2.239 | 0.735924 | 170.615311 | 71.54% | 29.938933 | 69.24% |
| 0.300 | 237 | 2.37% | 12 | 321 | 2,399 | 1.268 | 0.499344 | 112.357803 | 47.11% | 19.879691 | 45.98% |
| 0.500 | 225 | 2.25% | 0 | 333 | 8,040 | 1.000 | 0.406952 | 92.585080 | 38.82% | 16.507206 | 38.18% |

The unnormalized probability-weighted distance-deficit total was 129.713835; its
retained percentages equal those in the final column because normalization by the fixed
3.0 m threshold is constant. Full values are retained in
`probabilistic_risk_threshold_summary.csv`.

## Risk-proxy distributions

`M_unsafe` is QCNet probability mass assigned to modes below the screening threshold;
it is not calibrated collision probability. `R_expected` is normalized expected
distance deficit; it is not physical expected collision severity.

| Score / population | N | Mean | Median | P25 | P75 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M_unsafe`, all | 10,000 | 0.023849 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.002022 | 1.000000 |
| `M_unsafe`, positive | 558 | 0.427405 | 0.244998 | 0.022464 | 0.949438 | 1.000000 | 1.000000 | 1.000000 |
| `R_expected`, all | 10,000 | 0.004324 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000218 | 0.113189 |
| `R_expected`, positive | 558 | 0.077487 | 0.023167 | 0.002438 | 0.079375 | 0.202019 | 0.366400 | 0.785881 |

## Primary fixed-policy comparison

| Policy | Interventions | Rate | Positives | TP | FP / extra | FN | TN | Recall | Precision | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-1 | 225 | 2.25% | 159 | 119 | 106 | 40 | 9,735 | 0.748 | 0.529 | 0.011 | 0.620 |
| Worst-case | 558 | 5.58% | 159 | 141 | 417 | 18 | 9,424 | 0.887 | 0.253 | 0.042 | 0.393 |
| Probability-aware `theta=0.05` | 374 | 3.74% | 159 | 137 | 237 | 22 | 9,604 | 0.862 | 0.366 | 0.024 | 0.514 |
| Risk mass `rho=0.10` | 341 | 3.41% | 159 | 133 | 208 | 26 | 9,633 | 0.836 | 0.390 | 0.021 | 0.532 |

This table describes sensitivity/selectivity against the recorded AV2 future. It does
not establish a policy as optimal or prove safety improvement.

## Direct risk-policy sensitivity

The complete tie-preserving sweeps contain every observed score transition plus fixed
interpretable values: 588 risk-mass data rows and 603 expected-loss data rows (589 and
604 lines respectively when each CSV header is included).
The complete results are in `risk_mass_policy_sweep.csv` and
`expected_loss_policy_sweep.csv`. Selected predefined points are:

| Policy point | Interventions | Rate | TP | FP | FN | Recall | Precision | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rho=0.01` | 455 | 4.55% | 140 | 315 | 19 | 0.881 | 0.308 | 0.032 | 0.456 |
| `rho=0.05` | 381 | 3.81% | 137 | 244 | 22 | 0.862 | 0.360 | 0.025 | 0.507 |
| `rho=0.10` | 341 | 3.41% | 133 | 208 | 26 | 0.836 | 0.390 | 0.021 | 0.532 |
| `rho=0.20` | 295 | 2.95% | 130 | 165 | 29 | 0.818 | 0.441 | 0.017 | 0.573 |
| `rho=0.50` | 231 | 2.31% | 124 | 107 | 35 | 0.780 | 0.537 | 0.011 | 0.636 |
| `eta=0.01` | 337 | 3.37% | 123 | 214 | 36 | 0.774 | 0.365 | 0.022 | 0.496 |
| `eta=0.05` | 198 | 1.98% | 89 | 109 | 70 | 0.560 | 0.449 | 0.011 | 0.499 |
| `eta=0.10` | 114 | 1.14% | 43 | 71 | 116 | 0.270 | 0.377 | 0.007 | 0.315 |
| `eta=0.20` | 57 | 0.57% | 14 | 43 | 145 | 0.088 | 0.246 | 0.004 | 0.130 |
| `eta=0.50` | 19 | 0.19% | 3 | 16 | 156 | 0.019 | 0.158 | 0.002 | 0.034 |

No operating point was optimized on this validation cohort. The sweeps are secondary
sensitivity analyses around the frozen primary comparison.

## Score quality and reliability

The recorded-future event prevalence was 0.015900.

| Score | Brier | AUROC | AUPRC / average precision |
|---|---:|---:|---:|
| `M_unsafe` | 0.011465 | 0.936975 | 0.635297 |
| `R_expected` | Not applicable | 0.930743 | 0.333565 |

AUPRC must be interpreted relative to 1.59% prevalence. These exploratory metrics
measure discrimination against the recorded AV2 outcome reference; they do not prove
physical collision-risk calibration. No fitting or recalibration was performed.

Adaptive tie-preserving reliability bins for `M_unsafe` were:

| Score interval | N | Mean `M_unsafe` | Recorded-event frequency |
|---|---:|---:|---:|
| 0 exactly | 9,442 | 0.000000 | 0.001906 |
| (0, 0.077214] | 200 | 0.017581 | 0.030000 |
| [0.077696, approximately 1] | 358 | 0.656357 | 0.377095 |

Ties at zero and one were kept together, so three populated bins are more informative
than forcing ten bins with tiny or split tie groups.

## Commands

The following commands capture the substantive run. Timestamp/log wrappers are omitted
for readability; their outputs are preserved under `runpod_logs/` locally and
`/workspace/logs/qcnet_server_10000/` on RunPod.

```bash
PYTHONPATH=src python src/av_safety_eval/experiments/prepare_qcnet_validation_cohort.py \
  --raw-root /home/turan/ucl/data/argoverse2/val/raw \
  --historical-manifest results/qcnet_server_500/selected_scenario_ids.txt \
  --output-manifest results/qcnet_server_10000/selected_scenario_ids.txt \
  --summary-json results/qcnet_server_10000/integrity_check/cohort_selection_summary.json \
  --selected-count 10000 --seed 42
```

```bash
rsync -a --no-owner --no-group --partial --files-from=/tmp/qcnet_server_10000_av2_files.txt \
  -e 'ssh -p 22010' /home/turan/ucl/data/argoverse2/val/raw/ \
  root@RUNPOD:/workspace/data/argoverse2/server_10000/val/raw/
```

```bash
cd /workspace/external_repos/QCNet
QCNET_TINY_DATASET=1 PYTHONPATH=/workspace/external_repos/QCNet \
/workspace/envs/qcnet/bin/python -u -c \
  "from datasets import ArgoverseV2Dataset; d=ArgoverseV2Dataset(root='/workspace/data/argoverse2/server_10000', split='val'); print(len(d))"
```

```bash
cd /workspace/external_repos/QCNet
QCNET_TINY_DATASET=1 PYTHONPATH=/workspace/external_repos/QCNet:/workspace/scenario-safety-av/src \
/workspace/envs/qcnet/bin/python -u \
  /workspace/scenario-safety-av/scripts/export_qcnet_batch_scenario_artifacts.py \
  --root /workspace/data/argoverse2/server_10000 \
  --ckpt_path /workspace/models/qcnet/QCNet_AV2.ckpt \
  --output_dir /workspace/scenario-safety-av/results/qcnet_server_10000/artifacts \
  --scenario_ids_file /workspace/scenario-safety-av/results/qcnet_server_10000/selected_scenario_ids.txt \
  --expected_num_scenarios 10000 --device cuda
```

```bash
cd /workspace/scenario-safety-av
PYTHONPATH=src /workspace/envs/qcnet/bin/python \
  src/av_safety_eval/experiments/validate_qcnet_artifact_cohort.py \
  --artifact-dir results/qcnet_server_10000/artifacts \
  --manifest results/qcnet_server_10000/selected_scenario_ids.txt \
  --historical-manifest results/qcnet_server_500/selected_scenario_ids.txt \
  --expected-count 10000 \
  --output-dir results/qcnet_server_10000/integrity_check \
  --export-summary-csv results/qcnet_server_10000/artifacts/batch_export_summary.csv
```

```bash
PYTHONPATH=src /workspace/envs/qcnet/bin/python \
  src/av_safety_eval/experiments/evaluate_qcnet_batch_artifacts.py \
  --artifact-dir results/qcnet_server_10000/artifacts \
  --output-csv results/qcnet_server_10000/qcnet_server_10000_ranking.csv \
  --output-json results/qcnet_server_10000/qcnet_server_10000_ranking.json

PYTHONPATH=src MPLCONFIGDIR=/workspace/cache/matplotlib /workspace/envs/qcnet/bin/python \
  src/av_safety_eval/experiments/analyze_qcnet_probabilistic_risk.py \
  --artifact-dir results/qcnet_server_10000/artifacts \
  --selected-scenario-ids results/qcnet_server_10000/selected_scenario_ids.txt \
  --output-dir results/qcnet_server_10000/probabilistic_risk

PYTHONPATH=src MPLCONFIGDIR=/workspace/cache/matplotlib /workspace/envs/qcnet/bin/python \
  src/av_safety_eval/experiments/analyze_qcnet_risk_aware_decisions.py \
  --artifact-dir results/qcnet_server_10000/artifacts \
  --selected-scenario-ids results/qcnet_server_10000/selected_scenario_ids.txt \
  --output-dir results/qcnet_server_10000/risk_aware_decision
```

No `--reproduction-profile` was supplied for the independent cohort. The strict
`historical_500` profile remains available and was regression-tested against the exact
500 / 31 / 13 / 8 / 18 fingerprint.

## Output inventory and backup

Primary results are under `results/qcnet_server_10000/`:

- `selected_scenario_ids.txt`
- `qcnet_server_10000_ranking.csv` and `.json`
- `integrity_check/cohort_selection_summary.json`
- `integrity_check/integrity_summary.md`, `.json`, and `integrity_details.csv`
- `probabilistic_risk/probabilistic_risk_per_scenario.csv`
- `probabilistic_risk/probabilistic_risk_threshold_summary.csv`
- `probabilistic_risk/probabilistic_risk_summary.md`
- `probabilistic_risk/probabilistic_risk_scatter.png`
- `probabilistic_risk/probabilistic_risk_threshold_retention.png`
- `risk_aware_decision/risk_aware_per_scenario.csv`
- `risk_aware_decision/risk_mass_policy_sweep.csv`
- `risk_aware_decision/expected_loss_policy_sweep.csv`
- `risk_aware_decision/existing_policy_realized_outcomes.csv`
- `risk_aware_decision/fixed_policy_comparison.csv`
- `risk_aware_decision/risk_score_quality.csv`
- `risk_aware_decision/risk_aware_decision_summary.md`
- `risk_aware_decision/risk_decision_tradeoff.png`
- `risk_aware_decision/unsafe_mass_reliability.png`

The exact 10,000 artifacts, compact outputs, figures, source changes, and execution logs
are preserved both on RunPod under `/workspace` and on the local development machine.
Artifacts and logs are intentionally ignored by Git; reviewable source, manifests,
summaries, tables, figures, and this audit document are intended for version control.
The RunPod pod was not terminated.

## Remaining caveats

- Center-to-center distance is a screening metric, not exact actor-envelope geometry.
- QCNet mode weights are not calibrated physical collision probabilities.
- Only one future is recorded in AV2; plausible unrealized modes cannot be labelled
  simply correct or incorrect from this reference.
- AUROC, AUPRC, Brier score, and reliability bins describe correspondence to the
  recorded threshold event, with low 1.59% prevalence.
- `theta=0.05` and `rho=0.10` are predefined interpretable operating points, not
  validation-optimized thresholds.
- The analysis is open loop. Controlled CARLA experiments are separate closed-loop
  validation and do not run QCNet online.
- The independent cohort is large and disjoint from the historical 500, but it remains
  one deterministic random subset of the available AV2 validation scenarios.
