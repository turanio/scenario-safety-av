# highD Dataset Requirements

Checked on 2026-07-05.

## Access

highD must be requested manually from https://levelxdata.com/highd-dataset/. The dataset page describes highD as naturalistic highway trajectories from German highways, recorded by drone, with more than 110,500 vehicles. It says the dataset is free of charge for academic and research purposes, but each request is checked manually.

The terms shown on the dataset page restrict redistribution. Dataset files, modified dataset files, and raw excerpts must not be committed to this repository. Local highD data should live under an ignored path such as `data/highD/` or outside the thesis repo.

## Original highD Files

The official highD format description states that each of the 60 recordings has four files:

- `XX_highway.jpg`
- `XX_recordingMeta.csv`
- `XX_tracksMeta.csv`
- `XX_tracks.csv`

The recording metadata includes fields such as recording id, frame rate, location id, speed limit, duration, vehicle counts, and lane marking positions.

The track metadata includes one row per vehicle track with fields such as id, dimensions, initial/final frame, class, driving direction, distance traveled, min/max/mean x velocity, minimum DHW/THW/TTC, and number of lane changes.

The time-dependent tracks file includes frame, id, position, velocity, acceleration, headway/time-to-collision values, surrounding vehicle ids, and lane id.

## highD Columns Needed for This Thesis

At minimum, the thesis integration will need:

- `frame`
- `id`
- `x`
- `y`
- `width`
- `height`
- `xVelocity`
- `yVelocity`
- `xAcceleration`
- `yAcceleration`
- `precedingId`
- `followingId`
- `leftPrecedingId`
- `leftAlongsideId`
- `leftFollowingId`
- `rightPrecedingId`
- `rightAlongsideId`
- `rightFollowingId`
- `laneId`

Useful filtering and validation columns from `XX_tracksMeta.csv` include:

- `initialFrame`
- `finalFrame`
- `numFrames`
- `class`
- `drivingDirection`
- `minDHW`
- `minTHW`
- `minTTC`
- `numLaneChanges`

## cVMD Preprocessed Folder Structure

The cVMD README does not consume raw highD CSV files directly in the documented training commands. It expects a preprocessed MATLAB-file scenario layout:

```text
data/highD/
  train/
    class0/
      kl0.mat
    class1/
      lcr0.mat
    class2/
      lcl0.mat
  test/
    class0/
    class1/
    class2/
```

The documented class meanings are:

- `class0`: keep-lane scenarios
- `class1`: lane-change-right scenarios
- `class2`: lane-change-left scenarios

The `.mat` scenario files are expected to contain observed surrounding-vehicle histories and the target future. The README lists keys including:

- `data_keys`
- `observed_data_x`
- `observed_data_y`
- `observed_data_vx`
- `observed_data_vy`
- `scenario_type`
- `predicted_x`
- `predicted_y`
- `predicted_ax`
- `predicted_dpsi`
- `psi_0`
- `v0`

The cVMD README describes `N` as the maximum number of considered vehicles including ego, `T_o` as observation steps, and `T_p` as prediction steps. It gives the example of 3 seconds observed at 25 Hz and 5 seconds predicted at 25 Hz.

## Preprocessing Gap

The first real integration task is not model loading. It is converting raw highD CSV tracks into cVMD-compatible `.mat` scenario files and then into this repository's `AgentState`/`PredictionSet` abstractions.

Required decisions:

- Ego/target selection policy for lane-change and cut-in cases.
- Coordinate convention conversion from highD image-coordinate SI units into the planner's ego-centric or world frame.
- Observation horizon and prediction horizon.
- Train/test split by recording or scenario.
- Mapping of highD lane-change labels into `class0`, `class1`, and `class2`.

## Storage Policy

Keep only code, docs, and small synthetic examples in git. Keep highD data local and ignored:

```text
data/highD/
../external_repos/
```

No highD raw files, derived `.mat` files, cVMD checkpoints, or large result pickles should be committed.
