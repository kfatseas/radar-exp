"""Split-view visualisation of point cloud and spectrum data.

This function produces a two-panel plot comparing the discrete point
cloud representation of detected objects against the continuous
range–Doppler spectrum.  Bounding boxes are drawn around each object
on the spectrum to emphasise the localisation of detections.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, PowerNorm
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
    # --- early exits (unchanged functionality) ---
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

    # --- helpers (visual only) ---
    def _polish_axes(ax):
        ax.set_facecolor("#fbfbfd")
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.3)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", labelsize=11, length=6, width=0.8)
        ax.tick_params(axis="both", which="minor", length=3, width=0.6)
        for spine in ax.spines.values():
            spine.set_alpha(0.25)

    def _scatter_glow(ax, x, y, colors, base_size=60, layers=3):
        # Aura/glow with per-point colors (speed-based)
        for n in range(layers, 0, -1):
            ax.scatter(
                x, y,
                s=base_size * (1 + n * 0.7),
                c=colors,
                alpha=0.07 * n,  # slightly stronger than before
                linewidths=0,
                zorder=1,
            )

    def _get_velocity(p):
        # Prefer physical radial velocity if available; fallback to bin value
        for attr in ('radial_velocity', 'velocity', 'v', 'velocity_bin'):
            if hasattr(p, attr):
                return getattr(p, attr)
        return 0.0

    def _get_rcs(p):
        # Robust: try common RCS fields, fallback to SNR, then 1.0
        for attr in ('rcs', 'rcs_db', 'rcs_dbsm', 'snr'):
            if hasattr(p, attr):
                return getattr(p, attr)
        return 1.0

    # Colours per object (USED FOR POINT FILL ONLY)
    obj_colours = plt.cm.tab10(np.linspace(0, 1, len(objs)))

    # --- mappings for aura (speed) & size (RCS) ---
    # Gather all speeds to set a robust normalization that "pops"
    all_speeds = np.array(
        [abs(_get_velocity(p)) for obj in objs for p in obj.points],
        dtype=float
    ) if any(obj.points for obj in objs) else np.array([0.0])

    # Robust vmax: 95th percentile to avoid tiny dynamic range; ensure > 0
    if all_speeds.size > 1:
        speed_vmax = float(np.percentile(all_speeds, 95))
        if speed_vmax <= 0:
            speed_vmax = float(np.max(all_speeds))
    else:
        speed_vmax = float(all_speeds.max())
    if speed_vmax <= 0:
        speed_vmax = 1.0

    # Power-law boost for more drama in the low/mid range
    speed_norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=speed_vmax)
    speed_cmap = plt.cm.turbo  # vivid & high-contrast

    # RCS → size (robust scaling)
    all_rcs_vals = np.array([_get_rcs(p) for obj in objs for p in obj.points], dtype=float) \
                   if any(obj.points for obj in objs) else np.array([1.0])
    rcs_p5, rcs_p95 = (np.percentile(all_rcs_vals, 5), np.percentile(all_rcs_vals, 95)) \
                      if all_rcs_vals.size > 1 else (1.0, 1.0)
    rcs_span = max(rcs_p95 - rcs_p5, 1e-6)
    def _size_from_rcs(rcs_vals, min_size=28.0, max_size=115.0):
        vals = np.asarray(rcs_vals, dtype=float)
        if vals.size == 0:
            return np.array([])
        norm = np.clip((vals - rcs_p5) / rcs_span, 0.0, 1.0)
        return min_size + norm * (max_size - min_size)

    # --- figure layout ---
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12.5, 6.8), dpi=170, gridspec_kw={'width_ratios': [1, 1]}
    )
    fig.patch.set_facecolor("#ffffff")

    ax_left.set_title('Point Cloud', fontsize=16, fontweight='bold', pad=10)
    ax_right.set_title('Range–Doppler Spectrum', fontsize=16, fontweight='bold', pad=10)

    # -------------------------
    # Left plot: point cloud
    # -------------------------
    _polish_axes(ax_left)
    ax_left.set_axisbelow(True)

    for i, obj in enumerate(objs):
        if not obj.points:
            continue
        # Rotate point cloud 90 deg left (counterclockwise)
        xs = np.array([p.x for p in obj.points], dtype=float)
        ys = np.array([p.y for p in obj.points], dtype=float)
        xs_rot = -ys
        ys_rot = xs

        # Per-point mappings
        v_vals = np.array([_get_velocity(p) for p in obj.points], dtype=float)
        speeds = np.abs(v_vals)
        rcs_vals = np.array([_get_rcs(p) for p in obj.points], dtype=float)
        sizes = _size_from_rcs(rcs_vals)

        # Aura (per-point) based on speed magnitude
        aura_colors = speed_cmap(speed_norm(speeds))
        _scatter_glow(
            ax_left, xs_rot, ys_rot, aura_colors,
            base_size=max(54.0, float(np.mean(sizes) if sizes.size else 54.0)), layers=3
        )

        # Main points: FILL by object id colour; size by RCS; more transparent
        pts = ax_left.scatter(
            xs_rot, ys_rot,
            c=[obj_colours[i]],  # fill colour by object ID (legend-friendly)
            s=sizes,
            alpha=0.38,          # stronger transparency for density
            edgecolors=(0, 0, 0, 0.18),
            linewidth=0.5,
            label=f'Object {obj.label} ({len(obj)} pts)',
            zorder=2,
        )
        # Very light stroke to keep shape crisp without killing transparency
        pts.set_path_effects([pe.withStroke(linewidth=0.6, foreground="white", alpha=0.35)])

    ax_left.set_xlabel('X (m)', fontsize=13)
    ax_left.set_ylabel('Y (m)', fontsize=13)

    # Set limits (unchanged logic)
    if zoom_to_objects:
        xs = [p.x for obj in objs for p in obj.points]
        ys = [p.y for obj in objs for p in obj.points]
        xs_rot = [-y for y in ys]
        ys_rot = [x for x in xs]
        if xs_rot and ys_rot:
            x_margin = (max(xs_rot) - min(xs_rot)) * 0.15 + 5.0
            y_margin = (max(ys_rot) - min(ys_rot)) * 0.15 + 5.0
            ax_left.set_xlim(min(xs_rot) - x_margin, max(xs_rot) + x_margin)
            ax_left.set_ylim(0, max(ys_rot) + y_margin)  # positive-only Y
    else:
        lim = max_range
        ax_left.set_xlim(-lim, lim)
        ax_left.set_ylim(0, lim)

    ax_left.set_box_aspect(1)

    # Legend: same info (object counts), shows object fill colours
    leg = ax_left.legend(
        fontsize=11, frameon=True, fancybox=True, framealpha=0.92,
        edgecolor="#dddddd", borderpad=0.6, handlelength=1.6, loc="best"
    )
    leg.set_zorder(3)

    # -----------------------------------------
    # Right plot: RD spectrum with bboxes
    # -----------------------------------------
    _polish_axes(ax_right)
    n_range, n_doppler = rd_map.shape
    im = ax_right.imshow(
        rd_map, origin='lower', aspect='auto', cmap='inferno',
        extent=[0, n_doppler, 0, n_range], alpha=0.96, interpolation='bilinear', zorder=1
    )
    ax_right.set_ylim(0, n_range)
    ax_right.set_box_aspect(1)

    # Ticks with physical units (unchanged mapping)
    n_ticks = 5
    range_ticks = np.linspace(0, n_range, n_ticks)
    range_labels = [f"{r:.0f}" for r in np.linspace(0, max_range, n_ticks)]
    ax_right.set_yticks(range_ticks)
    ax_right.set_yticklabels(range_labels)

    vel_ticks = np.linspace(0, n_doppler, n_ticks)
    vel_labels = [f"{v:.0f}" for v in np.linspace(-max_velocity, max_velocity, n_ticks)]
    ax_right.set_xticks(vel_ticks)
    ax_right.set_xticklabels(vel_labels)

    ax_right.set_xlabel('Radial velocity (m/s)', fontsize=13)
    ax_right.set_ylabel('Range (m)', fontsize=13)

    ax_right.grid(True, alpha=0.35, color='white', linestyle='--', linewidth=0.6, zorder=2)

    # Draw bounding boxes: even slimmer; labels lighter
    for i, obj in enumerate(objs):
        if not obj.points:
            continue
        r_idxs = [p.r_idx for p in obj.points]
        d_idxs = [p.d_idx for p in obj.points]
        pad = 2
        r_min = max(0, min(r_idxs) - pad)
        r_max = min(n_range, max(r_idxs) + pad + 1)
        d_min = max(0, min(d_idxs) - pad)
        d_max = min(n_doppler, max(d_idxs) + pad + 1)

        rect = plt.Rectangle(
            (d_min, r_min), d_max - d_min, r_max - r_min,
            fill=False, edgecolor=obj_colours[i], linewidth=0.8, alpha=0.9, zorder=3
        )
        ax_right.add_patch(rect)

        ax_right.text(
            (d_min + d_max) / 2.0, r_max + 2.0, f'Obj {obj.label}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold',
            color=obj_colours[i], alpha=0.75, zorder=4,
            bbox=dict(boxstyle="round,pad=0.22", facecolor='white', edgecolor='none', alpha=0.42)
        )

    # Colourbar (spectrum)
    cbar = plt.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)
    cbar.set_label('Spectral Power (dB)', fontsize=12)
    cbar.outline.set_alpha(0.25)
    cbar.ax.tick_params(labelsize=10)

    # Layout & export (same filename)
    plt.tight_layout()
    plt.savefig('output.png', dpi=170, bbox_inches='tight', facecolor=fig.get_facecolor())
    print('Plot saved as output.png')

    # Print summary (unchanged content)
    print("\n=== Point Cloud vs Spectrum Visualization ===")
    n_points = sum(len(obj) for obj in objs)
    print(f"Left:  Point Cloud – {n_points} discrete detection points")
    print(f"Right: Spectrum – Continuous RD map with {len(objs)} object bounding boxes")
    print(f"Zoom: {'Enabled' if zoom_to_objects else 'Disabled'}")
