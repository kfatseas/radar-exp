from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.gridspec import GridSpec

from ..detect.pointcloud import Object
from ..detect.patches import get_patch_bounds, extract_patch


def gallery(
    rd_map: np.ndarray,
    objects: List[Object],
    padding: int = 2,
    max_cols: int = 4,
) -> None:
    """Display a gallery of objects (rows fully filled; multiple of 4 only)."""
    if not objects:
        print("No objects to visualise in gallery")
        return

    objs = [o for o in objects if getattr(o, "points", None)]
    if not objs:
        print("No objects with points to visualise in gallery")
        return

    # Enforce multiple-of-4 by dropping smallest objects
    n = len(objs)
    drop = n % 4
    if drop:
        objs = sorted(objs, key=lambda o: len(o.points))[drop:]
        print(f"Dropping {drop} smallest object(s) to make a multiple of 4 ({len(objs)} shown).")

    n = len(objs)
    if n == 0:
        print("No objects to visualise after applying multiple-of-4 rule")
        return

    # Choose column count that divides n (<= max_cols)
    max_candidate = min(n, max_cols)
    cols = next((c for c in range(max_candidate, 0, -1) if n % c == 0), min(max_cols, 4, n))
    rows = n // cols

    # --- Figure & grid (manual spacing; no tight/constrained layout) ---
    fig_w, fig_h = 5.2 * cols, 6.3 * rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)

    # More vertical spacing between cells so spectrum sits clearly below scatter
    outer = GridSpec(rows, cols, figure=fig, wspace=0.26, hspace=0.36)

    def _polish_axes(ax):
        ax.set_facecolor("#fbfbfd")
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.3)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.22)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", labelsize=9.5, length=5, width=0.8)
        ax.tick_params(axis="both", which="minor", length=3, width=0.6)
        for s in ax.spines.values():
            s.set_alpha(0.25)

    obj_colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    def _adaptive_size(num_pts: int) -> float:
        # More points -> smaller markers (bounded)
        return float(np.clip(110 - 0.04 * num_pts, 28, 90))

    for k, obj in enumerate(objs):
        r, c = divmod(k, cols)

        # Taller gap between the two rows inside each cell; RD patch nudged downward
        cell = outer[r, c].subgridspec(2, 1, height_ratios=[3.2, 2.1], hspace=0.20)

        # ----- top: Point Cloud -----
        ax_pc = fig.add_subplot(cell[0, 0])
        _polish_axes(ax_pc)

        xs = np.array([p.x for p in obj.points], dtype=float)
        ys = np.array([p.y for p in obj.points], dtype=float)

        s_size = _adaptive_size(len(obj.points))
        sc = ax_pc.scatter(
            xs, ys,
            s=s_size,
            c=[obj_colors[k]],
            alpha=0.78,
            edgecolors=(0, 0, 0, 0.18),
            linewidths=0.5,
            zorder=2,
            label=f"Obj {obj.label} ({len(obj.points)} pts)",
        )
        sc.set_path_effects([pe.withStroke(linewidth=0.7, foreground="white", alpha=0.45)])

        ax_pc.set_aspect('equal', adjustable='box')
        if xs.size and ys.size:
            xpad = (xs.max() - xs.min()) * 0.18 + 1.2
            ypad = (ys.max() - ys.min()) * 0.18 + 1.2
            xmin, xmax = (xs.min() - xpad, xs.max() + xpad) if xs.max() != xs.min() else (xs[0] - 2.0, xs[0] + 2.0)
            ymin, ymax = (ys.min() - ypad, ys.max() + ypad) if ys.max() != ys.min() else (ys[0] - 2.0, ys[0] + 2.0)
            ax_pc.set_xlim(xmin, xmax)
            ax_pc.set_ylim(ymin, ymax)

        ax_pc.set_title(f"Object {obj.label} — Point Cloud", fontsize=12, fontweight='bold', pad=6)
        ax_pc.set_xlabel("X (m)", fontsize=10.5)
        ax_pc.set_ylabel("Y (m)", fontsize=10.5)
        ax_pc.legend(frameon=True, fancybox=True, framealpha=0.92, fontsize=9, loc="best")

        # ----- bottom: RD patch -----
        ax_sp = fig.add_subplot(cell[1, 0])
        _polish_axes(ax_sp)

        bounds = get_patch_bounds(obj.points, rd_map.shape, padding)
        patch = extract_patch(rd_map, bounds)  # (rows, cols)
        pr, pc = patch.shape[:2]
        ax_sp.set_box_aspect((pr / pc) if pc > 0 else 1.0)  # correct pixel aspect

        ax_sp.imshow(
            patch,
            origin='lower',
            aspect='auto',          # box_aspect enforces pixel shape visually
            cmap='inferno',
            interpolation='nearest'
        )

        ax_sp.set_title("RD Patch", fontsize=11, pad=4)
        ax_sp.set_xlabel("Doppler bin", fontsize=10.5)
        ax_sp.set_ylabel("Range bin", fontsize=10.5)
        for spine in ax_sp.spines.values():
            spine.set_linewidth(0.8)
            spine.set_alpha(0.35)

    # Manual spacing to avoid layout solver clashes; leaves room below for patches
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.06, wspace=0.26, hspace=0.36)

    out_path = "output_gallery.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())  # no legend/colorbar; no tight bbox
    print(f"Gallery saved to {out_path}")
