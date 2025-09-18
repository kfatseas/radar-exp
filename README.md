## Radar Experiment Toolkit

This repository provides a modular framework for conducting experiments with
automotive radar data.  The goal of the project is to quantify the
information contained in range–Doppler (RD) spectra versus sparse point
clouds for object classification tasks.  It offers utilities for
pre‑processing radar cubes, performing CFAR based detection, extracting
patches and point clouds, computing simple feature vectors, training
classifiers, visualising results and sweeping over parameter grids.

### Quickstart

1. **Install dependencies**

   Use the provided `requirements.txt` to set up a Python environment.  The
   project depends only on numpy, scipy, matplotlib, scikit‑learn and
   pyyaml.

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare a recording**

   The framework operates on `src.recording.Recording` objects from your
   radar dataset.  It does not make assumptions about the underlying
   implementation; it only expects `cube(frame_idx)` to return a radar
   cube of shape `(num_antennas, num_chirps, num_samples)` and
   `settings` to expose the keys `'max distance'` and `'max velocity'`.

   Update `config/base.yaml` with the path to your recording.  Example:

   ```yaml
   dataset:
     path: "/path/to/your/recording"
     frame_indices: [0]  # frames to process
   ```

3. **Run a single experiment**

   The CLI runner loads the configuration, processes the specified
   frames, extracts RD patches and point clouds, computes features and
   writes results to a timestamped run directory.  Use the following
   command:

   ```bash
   python -m radar_exp.exp.runner --config config/base.yaml
   ```

   Outputs (e.g. metrics, parameters, plots) are saved under
   `runs/<timestamp>_<shortname>/`.

4. **Sweep over CFAR parameters**

   To evaluate the effect of different CFAR settings, define an ablation
   configuration under `config/ablations/`.  For example,
   `config/ablations/cfar_sweep.yaml` sweeps over CFAR guard and reference
   cell sizes as well as percentile thresholds.  Run the ablation via:

   ```bash
   python -m radar_exp.exp.ablate --config config/ablations/cfar_sweep.yaml
   ```

   Each sweep entry calls the runner with an overridden set of
   parameters and logs the results in a registry file.

5. **Produce split‑view visualisations**

   To compare RD maps with point clouds for a handful of detected
   objects, use the visualiser:

   ```bash
   from radar_exp.viz.rd_pc_split import visualize_split
   from radar_exp.io.recording_adapter import RecordingAdapter
   from radar_exp.dsp.rd_map import compute_range_doppler_map
   from radar_exp.detect.pointcloud import create_points, cluster_points

   rec = RecordingAdapter("/path/to/recording")
   rd_map, rd_cube = compute_range_doppler_map(rec.get_frame(0))
   points = create_points(rd_cube, rd_map, rec.max_range(), rec.max_velocity())
   objects = cluster_points(points)
   visualize_split(rd_map, objects)
   ```

### Configuration

The framework uses plain YAML configuration files.  A top‑level config
consists of the following sections:

- `dataset`: path to the recording and frames to process.
- `dsp`: window types, FFT sizes and aggregation modes for computing RD
  maps.
- `cfar`: method (`percentile` or `ca`), percentile threshold or guard
  and reference cell sizes for CFAR.
- `detect`: DBSCAN parameters `eps` and `min_samples` for clustering.
- `patch`: bounding box padding (in RD bins) and optional cropping.
- `pointcloud`: parameters controlling the DoA estimation (FFT size).
- `features`: toggles to enable RD and PC feature extraction.
- `model`: choice of classifier (`logistic`, `svm` or `random_forest`) and
  its hyper‑parameters.
- `metrics`: which evaluation metrics and information‑loss metrics to
  compute.

Fully worked examples can be found in `config/base.yaml` and
`config/ablations/cfar_sweep.yaml`.  Comments inside these files explain
each option.

### Migration map

This repository refactors the monolithic script in
`radar_object_detection.py` into a modular package.  The table below
summarises where the original functions and classes can be found now:

| Old function/class                    | New module & function                     |
|--------------------------------------|-------------------------------------------|
| `compute_range_fft`                  | `radar_exp.dsp.fft.range_fft`             |
| `compute_velocity_fft`               | `radar_exp.dsp.fft.velocity_fft`          |
| `compute_range_doppler_map`          | `radar_exp.dsp.rd_map.compute_range_doppler_map` |
| `detect_objects_in_rd_map`           | `radar_exp.detect.peaks.detect_peaks` (CFAR/threshold) |
| `extract_point_clouds`               | `radar_exp.detect.pointcloud.create_points` |
| `visualize_detections_and_point_clouds` | `radar_exp.viz.rd_pc_split.visualize_split` |
| `estimate_doa`                       | `radar_exp.detect.pointcloud.estimate_doa` |
| `Point` class                        | `radar_exp.detect.pointcloud.Point` (dataclass) |
| `Object` class                       | `radar_exp.detect.pointcloud.Object` (dataclass) |

To reproduce the plot created by `plot_rd_simple.py`, call the
visualiser from the new API as shown in the quickstart above.  The
default settings mirror the behaviour of the original script: a Hann
window, magnitude RD map aggregated over antennas by maximum, a 99th
percentile threshold for detection, basic DBSCAN clustering and a
two‑panel split view.

### Example commands

Run a single experiment using the base configuration:

```bash
python -m radar_exp.exp.runner --config config/base.yaml
```

Sweep CFAR parameters defined in `cfar_sweep.yaml`:

```bash
python -m radar_exp.exp.ablate --config config/ablations/cfar_sweep.yaml
```

Export a CSV summary of information‑loss metrics and classification
accuracy after an ablation run (assuming the registry file was written to
`runs/registry.csv`):

```bash
import pandas as pd
df = pd.read_csv('runs/registry.csv')
summary = df[['run_id','info_energy_retention','accuracy']]
summary.to_csv('runs/summary.csv', index=False)
```
