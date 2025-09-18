"""CLI runner for radar experiments.

This module defines a function `run_experiment` that processes one or
more frames from a radar recording according to a YAML configuration.
The high level steps are:

1. Load the radar cube(s) via `RecordingAdapter`.
2. Compute the range–Doppler map using configured DSP parameters.
3. Detect peaks via CFAR or percentile thresholding and clean the mask.
4. Convert detections into points with DoA estimation and physical
   coordinates.
5. Cluster points into objects using DBSCAN.
6. Extract RD patches and compute RD and PC features.
7. Compute information‑loss metrics.
8. Aggregate metrics across frames and objects and save results.

The runner writes its outputs to a timestamped directory under
`runs/`.  It saves the parameters, a CSV of per‑object results, and
aggregated metrics in JSON format.  Optionally it can train a
classifier when labels are supplied in the configuration.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
import pandas as pd

from ..io.recording_adapter import RecordingAdapter
from ..dsp.rd_map import compute_range_doppler_map
from ..detect.peaks import detect_peaks
from ..detect.pointcloud import create_points
from ..detect.cluster import cluster_points
from ..detect.patches import get_patch_bounds, extract_patch
from ..features.rd_features import extract_rd_features
from ..features.pc_features import extract_pc_features
from ..metrics.info_loss import compute_info_loss_metrics
from ..models.classifiers import train_classifier, predict
from ..metrics.common import compute_classification_metrics


def _override(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively override keys in a configuration dictionary.

    Keys in `overrides` can be dot separated to access nested
    dictionaries.  A new dictionary is returned; the input is not
    modified.
    """
    import copy
    result = copy.deepcopy(cfg)
    for key, value in overrides.items():
        parts = key.split('.')
        d = result
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value
    return result


def run_experiment(config: Dict[str, Any], run_name: str | None = None, output_root: str = "runs") -> Dict[str, Any]:
    """Execute a single radar experiment as specified by `config`.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration for the experiment.
    run_name : str, optional
        Short identifier for the run.  If not provided, uses the value
        of `config.get('name', 'exp')`.
    output_root : str, optional
        Directory under which to create the run directory.  Defaults
        to `'runs'`.

    Returns
    -------
    dict
        Summary metrics aggregated over all objects and frames.  Keys
        include `info_energy_retention_mean`, `info_sparsity_mean` and
        others.  The returned dictionary also contains the path to the
        run directory under the key `run_dir`.
    """
    # Determine run name
    if run_name is None:
        run_name = str(config.get("name", "exp"))
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save configuration
    with open(run_dir / "params.yaml", "w") as f:
        yaml.dump(config, f)
    # Fix random seed for reproducibility
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    # Load recording
    dataset_cfg = config.get("dataset", {})
    rec_path = dataset_cfg.get("path")
    if rec_path is None:
        raise ValueError("dataset.path must be specified in configuration")
    rec = RecordingAdapter(rec_path)
    frames = dataset_cfg.get("frame_indices")
    if frames is None:
        frames = list(range(rec.n_frames))
    # Extract DSP parameters
    dsp_cfg = config.get("dsp", {})
    windows_cfg = dsp_cfg.get("windows", {})
    fft_sizes = dsp_cfg.get("fft", {})
    aggregate = dsp_cfg.get("aggregate", "max")
    magnitude = dsp_cfg.get("magnitude", "db")
    # CFAR and postproc
    cfar_cfg = config.get("cfar", {})
    post_cfg = config.get("postproc", {})
    # Detection clustering
    detect_cfg = config.get("detect", {})
    eps = float(detect_cfg.get("eps", 1.0))
    min_samples = int(detect_cfg.get("min_samples", 5))
    # Patch padding
    patch_cfg = config.get("patch", {})
    padding = int(patch_cfg.get("padding", 2))
    # Point cloud and DoA
    pc_cfg = config.get("pointcloud", {})
    doa_fft_size = int(pc_cfg.get("fft_size", 181))
    # Feature toggles
    feat_cfg = config.get("features", {})
    use_rd_feats = bool(feat_cfg.get("rd_features", True))
    use_pc_feats = bool(feat_cfg.get("pc_features", True))
    # Metrics toggles
    metrics_cfg = config.get("metrics", {})
    compute_info = bool(metrics_cfg.get("compute_info_loss", True))
    # Model configuration (optional)
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type")
    model_params = model_cfg.get("params", {}) or {}
    # Storage for per‑object results
    rows: List[Dict[str, Any]] = []
    # Process frames
    for frame_idx in frames:
        cube = rec.get_frame(frame_idx)
        # Compute RD map and cube
        rd_map, rd_cube = compute_range_doppler_map(
            cube,
            windows=windows_cfg,
            fft_sizes=fft_sizes,
            aggregate=aggregate,
            magnitude=magnitude,
        )
        # Detect peaks
        coords, mask = detect_peaks(rd_map, cfar_cfg, post_cfg)
        # Create points
        points = create_points(rd_cube, rd_map, rec.max_range(), rec.max_velocity(), coords, doa_fft_size=doa_fft_size)
        # Cluster into objects
        objects = cluster_points(points, rd_map.shape, eps=eps, min_samples=min_samples)
        # Sort objects by size
        objects = sorted(objects, key=len, reverse=True)
        # Extract features and metrics
        for obj in objects:
            # Extract patch
            bounds = get_patch_bounds(obj.points, rd_map.shape, padding)
            patch = extract_patch(rd_map, bounds)
            feats = {}
            if use_rd_feats:
                feats.update(extract_rd_features(patch))
            if use_pc_feats:
                feats.update(extract_pc_features(obj.points))
            info = {}
            if compute_info:
                info.update(compute_info_loss_metrics(patch, obj.points))
            row = {
                "frame": frame_idx,
                "object_label": obj.label,
            }
            row.update(feats)
            row.update(info)
            rows.append(row)
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    # Save per‑object results
    csv_path = run_dir / "objects.csv"
    df.to_csv(csv_path, index=False)
    # Aggregate information‑loss metrics across all objects
    summary: Dict[str, float] = {}
    if not df.empty:
        for col in [c for c in df.columns if c.startswith("energy_retention") or c.startswith("sparsity") or c.startswith("entropy_diff")]:
            summary[f"{col}_mean"] = float(df[col].mean())
    # Classification (optional).  Requires `labels` column in df.
    class_metrics: Dict[str, float] = {}
    if model_type and "label" in df.columns:
        # Extract feature columns (exclude non‑features)
        feature_cols = [c for c in df.columns if c not in {"frame", "object_label", "label"} and not c.startswith("energy_retention") and not c.startswith("sparsity") and not c.startswith("entropy_diff")]
        X = df[feature_cols].to_numpy()
        y = df["label"].to_numpy()
        # Split into train/test using a simple shuffle split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        model = train_classifier(X_train, y_train, model_type=model_type, params=model_params)
        y_pred, y_prob = predict(model, X_test)
        class_metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        # Save predictions
        preds_df = pd.DataFrame({"true": y_test, "pred": y_pred})
        if y_prob is not None:
            # Save probability of positive class if binary
            if y_prob.shape[1] >= 2:
                preds_df["prob"] = y_prob[:, 1]
        preds_df.to_csv(run_dir / "preds.csv", index=False)
    # Save summary and classification metrics
    summary.update(class_metrics)
    summary["run_dir"] = str(run_dir)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a radar experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--name", type=str, default=None, help="Optional short name for the run")
    parser.add_argument("--output", type=str, default="runs", help="Root directory for output runs")
    args = parser.parse_args()
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    summary = run_experiment(cfg, run_name=args.name, output_root=args.output)
    print("Experiment completed. Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()