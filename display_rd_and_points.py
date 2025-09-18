"""
Display RD spectrum and point clouds of detected objects from a radar recording.

Usage:
    python display_rd_and_points.py /path/to/recording [frame_idx]

If frame_idx is not provided, defaults to 0.
"""
import sys
from radar_exp.io.recording_adapter import RecordingAdapter
from radar_exp.dsp.rd_map import compute_range_doppler_map
from radar_exp.detect.pointcloud import create_points
from radar_exp.detect.cluster import cluster_points
from radar_exp.detect.peaks import detect_peaks
from radar_exp.viz.rd_pc_split import visualize_split

if len(sys.argv) < 2:
    print("Usage: python display_rd_and_points.py /path/to/recording [frame_idx]")
    sys.exit(1)

rec_path = sys.argv[1]
frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

rec = RecordingAdapter(rec_path)

# Determine frame range
try:
    n_frames = rec.n_frames
except Exception:
    n_frames = 1

for frame_idx in range(n_frames):
    rd_map, rd_cube = compute_range_doppler_map(rec.get_frame(frame_idx))
    coords, _ = detect_peaks(rd_map)
    points = create_points(rd_cube, rd_map, rec.max_range(), rec.max_velocity(), coords)
    objects = cluster_points(points, rd_map.shape, eps=2.0, min_samples=15)
    # Save each frame as an image
    from radar_exp.viz.rd_pc_split import visualize_split
    visualize_split(rd_map, objects, zoom_to_objects=False, max_range=rec.max_range(), max_velocity=rec.max_velocity())
    import os
    os.rename('output.png', f'frames/frame_{frame_idx:03d}.png')
    print(f'Saved frame {frame_idx}')

# mkdir -p frames && ffmpeg -framerate 5 -i frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p output.mp4