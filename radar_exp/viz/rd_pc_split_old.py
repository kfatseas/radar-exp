"""Split‑view visualisation of point cloud and spectrum data.

This function produces a two‑panel plot comparing the discrete point
cloud representation of detected objects against the continuous
range–Doppler spectrum.  Bounding boxes are drawn around each object
on the spectrum to emphasise the localisation of detections.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from ..detect.pointcloud import Object


def visualize_split(
    rd_map: np.ndarray,
    objects: List[Object],
    selected_objects: Optional[Iterable[int]] = None,
    max_range: Optional[float] = None,
    max_velocity: Optional[float] = None,
    zoom_to_objects: bool = True,
) -> None:
    """Display a split view of point cloud versus RD spectrum.

    Parameters
    ----------
    rd_map : np.ndarray
        The range–Doppler map (shape `(range_bins, doppler_bins)`).
    objects : list of Object
        Clustered objects containing points with physical coordinates.
    selected_objects : iterable of int, optional
        Indices of objects to plot.  If `None`, all objects are shown.
    max_range : float, optional
        Maximum range in metres for axis scaling.  If not provided it
        will be estimated from the points.
    max_velocity : float, optional
        Maximum velocity in m/s for axis scaling.  If not provided it
        will be estimated from the points.
    zoom_to_objects : bool, optional
        If True, the point cloud axes are zoomed to the extents of the
        selected objects.  Otherwise a symmetric range based on
        `max_range` is used.
    """
    if not objects:
        print("No objects to visualize")
        return
    # Determine which objects to display
    if selected_objects is None:
        indices = list(range(len(objects)))
    else:
        indices = [i for i in selected_objects if 0 <= i < len(objects)]
    if not indices:
        print("No valid objects selected")
        return
    objs = [objects[i] for i in indices]
    # Estimate axis scales if not provided
    if max_range is None:
        all_ranges = [p.range_bin for obj in objs for p in obj.points]
        max_range = max(all_ranges) if all_ranges else 1.0
    if max_velocity is None:
        all_vels = [p.velocity_bin for obj in objs for p in obj.points]
        max_velocity = max(abs(v) for v in all_vels) if all_vels else 1.0
    # Generate colours
    colours = plt.cm.tab10(np.linspace(0, 1, len(objs)))
    # Make both plots have the same height in pixels on the screen
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
    # Left plot: point cloud
    ax_left.set_title('Point Cloud', fontsize=16, fontweight='bold')
    for i, obj in enumerate(objs):
        # Rotate point cloud 90 deg left (counterclockwise)
        xs = [p.x for p in obj.points]
        ys = [p.y for p in obj.points]
        xs_rot = [-y for y in ys]
        ys_rot = [x for x in xs]
        ax_left.scatter(xs_rot, ys_rot, c=[colours[i]], s=60, alpha=0.8,
                        edgecolors='black', linewidth=0.5,
                        label=f'Object {obj.label} ({len(obj)} pts)')
    ax_left.set_xlabel('X (m)', fontsize=14)
    ax_left.set_ylabel('Y (m)', fontsize=14)
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=11)
    # Set limits
    if zoom_to_objects:
        xs = [p.x for obj in objs for p in obj.points]
        ys = [p.y for obj in objs for p in obj.points]
        xs_rot = [-y for y in ys]
        ys_rot = [x for x in xs]
        if xs_rot and ys_rot:
            x_margin = (max(xs_rot) - min(xs_rot)) * 0.15 + 5.0
            y_margin = (max(ys_rot) - min(ys_rot)) * 0.15 + 5.0
            ax_left.set_xlim(min(xs_rot) - x_margin, max(xs_rot) + x_margin)
            # Limit y-axis to positive values only
            ax_left.set_ylim(0, max(ys_rot) + y_margin)
    else:
        lim = max_range
        ax_left.set_xlim(-lim, lim)
        ax_left.set_ylim(0, lim)
    # Force both axes to have the same box aspect ratio (height in pixels)
    ax_left.set_box_aspect(1)
    # Right plot: RD spectrum with bounding boxes
    ax_right.set_title('Range–Doppler Spectrum', fontsize=16, fontweight='bold')
    n_range, n_doppler = rd_map.shape
    im = ax_right.imshow(rd_map, origin='lower', aspect='auto', cmap='inferno',
                         extent=[0, n_doppler, 0, n_range], alpha=0.9)
    ax_right.set_ylim(0, n_range)
    # Force RD map plot to have the same box aspect ratio as point cloud
    ax_right.set_box_aspect(1)
    # Ticks with physical units
    # Range axis corresponds to vertical axis (rows)
    n_ticks = 5
    range_ticks = np.linspace(0, n_range, n_ticks)
    range_labels = [f"{r:.0f}" for r in np.linspace(0, max_range, n_ticks)]
    ax_right.set_yticks(range_ticks)
    ax_right.set_yticklabels(range_labels)
    # Doppler axis corresponds to horizontal axis (cols).  Map to velocity
    vel_ticks = np.linspace(0, n_doppler, n_ticks)
    vel_labels = [f"{v:.0f}" for v in np.linspace(-max_velocity, max_velocity, n_ticks)]
    ax_right.set_xticks(vel_ticks)
    ax_right.set_xticklabels(vel_labels)
    ax_right.set_xlabel('Radial velocity (m/s)', fontsize=14)
    ax_right.set_ylabel('Range (m)', fontsize=14)
    ax_right.grid(True, alpha=0.3, color='white')
    # Draw bounding boxes
    for i, obj in enumerate(objs):
        if not obj.points:
            continue
        r_idxs = [p.r_idx for p in obj.points]
        d_idxs = [p.d_idx for p in obj.points]
        # compute bounds with padding
        pad = 2
        r_min = max(0, min(r_idxs) - pad)
        r_max = min(n_range, max(r_idxs) + pad + 1)
        d_min = max(0, min(d_idxs) - pad)
        d_max = min(n_doppler, max(d_idxs) + pad + 1)
        rect = plt.Rectangle((d_min, r_min), d_max - d_min, r_max - r_min,
                             fill=False, edgecolor=colours[i], linewidth=2.5, alpha=0.9)
        ax_right.add_patch(rect)
        # label above box
        ax_right.text((d_min + d_max) / 2.0, r_max + 2.0,
                      f'Obj {obj.label}',
                      ha='center', va='bottom', fontsize=10, fontweight='bold',
                      color=colours[i],
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    # Colourbar
    cbar = plt.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)
    cbar.set_label('Spectral Power (dB)', fontsize=12)
    plt.tight_layout()
    plt.savefig('output.png')
    print('Plot saved as output.png')
    # Print summary
    print("\n=== Point Cloud vs Spectrum Visualization ===")
    n_points = sum(len(obj) for obj in objs)
    print(f"Left:  Point Cloud – {n_points} discrete detection points")
    print(f"Right: Spectrum – Continuous RD map with {len(objs)} object bounding boxes")
    print(f"Zoom: {'Enabled' if zoom_to_objects else 'Disabled'}")