"""
Display RD spectrum and point clouds of detected objects from a radar recording.

Usage:
    python display_rd_and_points.py --recording /path/to/recording --start START_IDX [--length N] [--gallery]

- When --length is provided, processes a sequence starting at START_IDX with N frames.
- When --length is omitted, processes just the single frame START_IDX.
"""
import os
import argparse
from radar_exp.io.recording_adapter import RecordingAdapter
from radar_exp.dsp.rd_map import compute_range_doppler_map
from radar_exp.detect.pointcloud import create_points
from radar_exp.detect.cluster import cluster_points
from radar_exp.detect.peaks import detect_peaks
from radar_exp.viz.rd_pc_split import visualize_split

def main():
    parser = argparse.ArgumentParser(description="Display RD spectrum and point clouds from radar recording.")
    parser.add_argument("--recording", type=str, required=True, help="Path to radar recording")
    parser.add_argument("--start", type=int, default=0, help="Starting frame index (default: 0)")
    parser.add_argument("--length", type=int, default=None, help="Number of frames to process from start; omit to process a single frame")
    parser.add_argument("--gallery", action="store_true", help="Also save gallery visualization for each frame")
    args = parser.parse_args()

    rec = RecordingAdapter(args.recording)
    try:
        n_frames = rec.n_frames
    except Exception:
        n_frames = 1

    # Create a subfolder for each recording
    rec_name = os.path.basename(os.path.normpath(args.recording))
    out_dir = os.path.join('output', 'frames', rec_name)
    os.makedirs(out_dir, exist_ok=True)

    # Determine frame range from start and optional length
    start_idx = max(0, min(args.start, max(n_frames - 1, 0)))
    count = args.length if args.length is not None else 1
    end_idx = max(start_idx, min(start_idx + max(0, count), n_frames))
    frame_range = range(start_idx, end_idx)

    from radar_exp.viz.gallery import gallery

    for frame_idx in frame_range:
        rd_map, rd_cube = compute_range_doppler_map(rec.get_frame(frame_idx))
        coords, _ = detect_peaks(rd_map)
        points = create_points(rd_cube, rd_map, rec.max_range(), rec.max_velocity(), coords)
        objects = cluster_points(points, rd_map.shape, eps=1.0, min_samples=10)

        # Split view
        visualize_split(rd_map, objects, zoom_to_objects=False, max_range=rec.max_range(), max_velocity=rec.max_velocity())
        split_path = os.path.join(out_dir, f'frame_{frame_idx:03d}_split.png')
        if os.path.exists('output.png'):
            os.replace('output.png', split_path)
        print(f'Saved split view for frame {frame_idx} to {split_path}')

        # Gallery view (only if --gallery is set)
        if args.gallery:
            import matplotlib.pyplot as plt
            gallery(rd_map, objects)
            plt.savefig('output.png')
            plt.close()
            gal_path = os.path.join(out_dir, f'frame_{frame_idx:03d}_gal.png')
            if os.path.exists('output.png'):
                os.replace('output.png', gal_path)
            print(f'Saved gallery view for frame {frame_idx} to {gal_path}')

    # Example: ffmpeg command to create video
    # ffmpeg -framerate 5 -i output/frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p output.mp4

if __name__ == "__main__":
    main()