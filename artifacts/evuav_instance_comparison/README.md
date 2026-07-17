# EVUAV Instance Comparison

This directory prepares point-aligned EVUAV test predictions for Grid Mamba,
CETUS, and EV-SpSegNet. Existing prediction files in the three model run
directories are not overwritten.

## Protocol

- Test data: the same 24 files, ordered from `test_000.npz` to `test_023.npz`.
- Point identity: `point_idx` is the original row index in the source NPZ.
- Coordinates: `x`, `y`, and `t` are copied from the source NPZ `ev_loc` array.
- Semantic thresholds: Grid Mamba `0.41`, CETUS `0.47`, EV-SpSegNet `0.47`.
- CETUS feature normalization: recomputed once from the EVUAV training split
  and frozen in `generated_configs/cetus_train_norm_stats.json`.
- Shared tracking: `time_bin_ms=100`, `max_link_gap_bins=3`,
  `max_link_distance_px=45`, and `cc_dilate_pixels=2`.

The model-specific thresholds are their fixed best-IoU operating points. The
subsequent trajectory grouping parameters are shared unchanged across models.

## Run

```bash
bash run_predictions.sh
```

Predictions are written to `outputs/predictions/<model>/predictions.txt`.
`outputs/prediction_validation.json` is produced only after every row has been
matched back to the original NPZ coordinates and semantic label.

The current inference scripts require an NVIDIA GPU. Configuration generation
can be run without a GPU:

```bash
conda run -n grid_mamba python prepare_test_configs.py
cd /home/jzw/experiments/CETUS
PYTHONPATH=src conda run -n grid_mamba python \
  /home/jzw/experiments/grid_mamba_stream/artifacts/evuav_instance_comparison/prepare_cetus_norm_stats.py
```
