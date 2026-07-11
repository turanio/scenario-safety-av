# Argoverse 2 Dataset Requirements

Checked on 2026-07-05.

## Required Split

For the first QCNet stage, use the Argoverse 2 Motion Forecasting validation split. It is the best smoke-test split because it has ground truth futures and can be used with QCNet validation code. The test split is useful for benchmark-style submission generation but is not the first target for planner safety analysis.

The full training split is not required for the first thesis integration if the released QCNet AV2 checkpoint is used.

## Dataset Size and Access

The Argoverse 2 website describes the Motion Forecasting Dataset as 250,000 scenarios. Each scenario is 11 seconds long, sampled at 10 Hz, and contains the 2D bird's-eye-view centroid and heading of tracked objects. The download page says the motion forecasting archives total 58 GB.

Argoverse 2 is free under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License. The AV2 API code is MIT licensed.

Do not commit AV2 data. Store it locally under an ignored path such as:

```text
data/argoverse2/
data/av2/
```

or outside this repository.

## Format and Fields

The Argoverse 2 motion forecasting guide describes each scenario with:

- `scenario_id`
- `timestamps_ns`
- `tracks`
- `focal_track_id`
- `city_name`

Each track has:

- `track_id`
- `object_states`
- `object_type`
- `category`

Each object state has:

- `observed`
- `timestep`
- `position`
- `heading`
- `velocity`

Track categories are used for challenge scoring and data quality:

- `TRACK_FRAGMENT`
- `UNSCORED_TRACK`
- `SCORED_TRACK`
- `FOCAL_TRACK`

Object taxonomy includes dynamic actor classes such as `VEHICLE`, `PEDESTRIAN`, `MOTORCYCLIST`, `CYCLIST`, and `BUS`, plus static/background categories.

## Horizons and Sampling

QCNet's documented Argoverse 2 command uses:

- `num_historical_steps=50`
- `num_future_steps=60`

At 10 Hz, this corresponds to 5 seconds of observed history and 6 seconds of future prediction. The thesis planner may downsample or resample later, but the first adapter should preserve the QCNet/AV2 timing.

## Maps

Argoverse 2 scenarios are paired with local HD maps. The map format includes lane geometry, lane boundaries, lane marking types, traffic direction, crosswalks, driveable areas, and intersection information. QCNet uses map context through AV2 and its own preprocessing pipeline.

For the first adapter milestone, maps should remain inside the QCNet preprocessing and model input path. The `PredictionSet` output only needs predicted actor trajectories and probabilities.

## QCNet Expected Folder Structure

QCNet's dataset code expects a dataset root with split-specific raw and processed folders. Its `ArgoverseV2Dataset` supports:

```text
/path/to/dataset_root/
  train/
    raw/
    processed/
  val/
    raw/
    processed/
  test/
    raw/
    processed/
```

The first time QCNet runs, preprocessing may create processed `.pkl` files. These files can be large and should remain ignored.

## Tiny Smoke Test

The smallest useful smoke test is:

1. Download or copy a tiny subset of AV2 validation scenarios.
2. Point QCNet at a root containing only that subset, if its dataset class accepts the reduced raw directory.
3. Run validation or direct model inference with batch size 1, low worker count, and the released checkpoint.
4. Save one scenario's mode trajectories and probabilities as a small derived artifact.
5. Convert that artifact into this repository's `PredictionSet` without importing QCNet in the main package.

## Scenario Selection for Safety Evaluation

AV2 is urban and interaction-rich, not a one-to-one highway cut-in dataset. Scenario mining should select risk patterns rather than exact synthetic duplicates.

Useful filters:

- close longitudinal gap between ego/focal/scored vehicle and another vehicle;
- lateral movement across lane centerlines;
- actor heading or lateral velocity changing over the observed/future window;
- merge-like approach into the same lane corridor;
- crossing conflict at intersections or crosswalks;
- low predicted or observed minimum distance;
- high interaction density near the ego vehicle.

Scenario families for this thesis:

- cut-in-like vehicle interaction;
- lane change;
- merge;
- crossing or intersection conflict;
- close interaction with small safety margin.
