"""
Export radar experiment data to a numpy .npz file.

This script loads a radar recording, processes a frame, and exports a numpy dataset
containing point cloud features, spectrum patch features, and object class labels (if available).
"""
import numpy as np
import yaml
import argparse
from pathlib import Path

from radar_exp.io.recording_adapter import RecordingAdapter
from radar_exp.dsp.rd_map import compute_range_doppler_map
from radar_exp.detect.peaks import detect_peaks
from radar_exp.detect.pointcloud import create_points
from radar_exp.detect.cluster import cluster_points
from radar_exp.detect.patches import get_patch_bounds, extract_patch
from radar_exp.features.rd_features import extract_rd_features
from radar_exp.features.pc_features import extract_pc_features


def main():
    parser = argparse.ArgumentParser(description="Export radar experiment data to numpy .npz")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to process")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    dataset_cfg = config.get("dataset", {})
    rec_path = dataset_cfg.get("path")
    if rec_path is None:
        raise ValueError("dataset.path must be specified in config")
    rec = RecordingAdapter(rec_path)

    # DSP params
    dsp_cfg = config.get("dsp", {})
    windows_cfg = dsp_cfg.get("windows", {})
    fft_sizes = dsp_cfg.get("fft", {})
    aggregate = dsp_cfg.get("aggregate", "max")
    magnitude = dsp_cfg.get("magnitude", "db")
    # Detection params
    cfar_cfg = config.get("cfar", {})
    post_cfg = config.get("postproc", {})
    detect_cfg = config.get("detect", {})
    eps = float(detect_cfg.get("eps", 1.0))
    min_samples = int(detect_cfg.get("min_samples", 5))
    # Patch params
    patch_cfg = config.get("patch", {})
    padding = int(patch_cfg.get("padding", 2))
    # Point cloud params
    pc_cfg = config.get("pointcloud", {})
    doa_fft_size = int(pc_cfg.get("fft_size", 181))

    # Process frame
    cube = rec.get_frame(args.frame)
    rd_map, rd_cube = compute_range_doppler_map(
        cube,
        windows=windows_cfg,
        fft_sizes=fft_sizes,
        aggregate=aggregate,
        magnitude=magnitude,
    )
    coords, mask = detect_peaks(rd_map, cfar_cfg, post_cfg)
    points = create_points(rd_cube, rd_map, rec.max_range(), rec.max_velocity(), coords, doa_fft_size=doa_fft_size)
    objects = cluster_points(points, rd_map.shape, eps=eps, min_samples=min_samples)
    objects = sorted(objects, key=len, reverse=True)

    # Prepare arrays
    pc_features_list = []
    rd_features_list = []
    class_labels = []
    for obj in objects:
        bounds = get_patch_bounds(obj.points, rd_map.shape, padding)
        patch = extract_patch(rd_map, bounds)
        pc_feats = extract_pc_features(obj.points)
        rd_feats = extract_rd_features(patch)
        pc_features_list.append([pc_feats[k] for k in sorted(pc_feats.keys())])
        rd_features_list.append([rd_feats[k] for k in sorted(rd_feats.keys())])
        # Use object label as class (or set to -1 if not available)
        class_labels.append(obj.label if hasattr(obj, "label") else -1)

    # Convert to numpy arrays
    pc_features_arr = np.array(pc_features_list)
    rd_features_arr = np.array(rd_features_list)
    class_labels_arr = np.array(class_labels)

    # Save to .npz
    np.savez(args.output,
             pc_features=pc_features_arr,
             rd_features=rd_features_arr,
             class_labels=class_labels_arr)
    print(f"Exported {len(objects)} objects to {args.output}")

if __name__ == "__main__":
    main()
