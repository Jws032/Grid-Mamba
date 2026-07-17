# EVUAV Trajectory-level Instance Grouping Visualization

This self-contained bundle renders paper-ready 2x2 XY projections for three
EVUAV test candidates. It does not require the original dataset, checkpoint,
or `grid_mamba_stream` source tree after copying.

## Directory Layout

Each sample is isolated in matching input and output subdirectories:

```text
inputs/<sample>/
  <sample>.npz
  predictions_tracks.txt
  source_manifest.json
outputs/<sample>/
  <sample>_xy_2x2_preview.png
  <sample>_xy_2x2.png
  <sample>_xy_2x2.pdf
  panel_stats.json
  color_mapping.csv
```

`outputs/candidate_summary.csv` compares all candidates and links each preview.
`outputs/candidate_contact_sheet.png` places the three previews in one selection
sheet with the category, instance counts, and ARI above each candidate.

## Candidates

| Sample | Category | GT / Pred | ARI | Main characteristic |
|---|---|---:|---:|---|
| `test_013` | clean | 7 / 7 | 1.0000 | Original main candidate |
| `test_014` | mild-error | 6 / 6 | 0.9971 | Near-perfect with one split/miss |
| `test_020` | challenging | 5 / 5 | 0.7865 | Split/merge error case |

## Reproduce

Create an environment with Python 3.9 or newer, then generate all candidates:

```bash
python -m pip install -r requirements.txt
bash run_visualization.sh
```

Generate or validate one sample:

```bash
bash run_visualization.sh --sample test_013
python visualize_evuav_track_panels.py --sample test_013 --validate-only
python visualize_evuav_track_panels.py --list-samples
```

Validation checks each bundled NPZ and prediction subset against its byte size
and SHA256 before checking row-level alignment and expected statistics. All
runtime paths are relative to this directory.

The semantic prediction is reconstructed from `prob >= 0.73`; the stored
`pred` column is retained only for provenance because it was produced under a
different decision threshold. Semantic foreground is shown in red, and the
background events use a darker gray with higher opacity. Predicted track
colors are matched to GT colors with one-to-one Hungarian maximum-overlap
matching. This changes display colors only and never changes `track_id`.
Every panel includes a compact upper-right legend. Instance panels keep the
raw GT and Track IDs, while unmatched tracks and filtered points are labeled
explicitly.

Paper PNGs use 1200 DPI. PDFs keep text and axes as vector elements while
rasterizing event-point collections to control file size.
