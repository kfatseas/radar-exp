## Migration Map

This document serves as a guide for developers porting code from the
original monolithic implementation (`radar_object_detection.py` and
`plot_rd_simple.py`) into the modular structure provided by this
package.  The following table lists the original functions and
classes and where they have been relocated in the new project.

| Original entity                         | New location                                 | Notes                                                      |
|-----------------------------------------|-----------------------------------------------|------------------------------------------------------------|
| `compute_range_fft`                     | `radar_exp.dsp.fft.range_fft`                | Now accepts window type and FFT size via config            |
| `compute_velocity_fft`                  | `radar_exp.dsp.fft.velocity_fft`             | Ditto; Doppler FFT size is configurable                    |
| `compute_range_doppler_map`             | `radar_exp.dsp.rd_map.compute_range_doppler_map` | Aggregates antennas by max or sum; returns map and cube    |
| `detect_objects_in_rd_map`              | `radar_exp.detect.peaks.detect_peaks`        | CFAR implementation replaces global percentile threshold    |
| `extract_point_clouds`                  | `radar_exp.detect.pointcloud.create_points`   | Constructs `Point` objects with physical coordinates       |
| `generate_points`                       | `radar_exp.detect.pointcloud.create_points`   | Combined with DoA estimation                               |
| `cluster_points`                        | `radar_exp.detect.cluster.cluster_points`     | Uses DBSCAN; clusters in (range, velocity, doa) space      |
| `visualize_detections_and_point_clouds` | `radar_exp.viz.rd_pc_split.visualize_split`   | Two‑panel point cloud vs RD spectrum visualisation         |
| `estimate_doa`                          | `radar_exp.detect.pointcloud.estimate_doa`    | DoA estimation remains based on antenna FFT                |
| `Point` class                           | `radar_exp.detect.pointcloud.Point` (dataclass) | Adds Cartesian coordinates `x` and `y`                     |
| `Object` class                          | `radar_exp.detect.pointcloud.Object` (dataclass) | Now carries patch mask and centroid computation           |
| `visualize_pointcloud_vs_spectrum`      | `radar_exp.viz.rd_pc_split.visualize_split`   | Name changed to emphasise general use                      |
| `plot_rd_simple.py` main                | Example usage shown in README under Quickstart | Script replaced by CLI runner (`python -m radar_exp.exp.runner`) |

To replicate the behaviour of the original simple test:

```bash
python -m radar_exp.exp.runner --config config/base.yaml
```

This will compute the range–Doppler map, perform detection, cluster points,
extract patches and compute features for the specified frames.  It will
save per‑object information and a summary of information‑loss metrics to
the `runs/` directory.
