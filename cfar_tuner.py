"""
Interactive CFAR tuner (fixed): responsive sizing, proper refresh, independent zoom.
"""
import holoviews as hv
import panel as pn
import numpy as np
import yaml
from pathlib import Path

hv.extension("bokeh")
pn.extension(sizing_mode="stretch_both")

# --- radar-exp imports ---
from radar_exp.io.recording_adapter import RecordingAdapter
from radar_exp.dsp.rd_map import compute_range_doppler_map
from radar_exp.detect.peaks import detect_peaks
from radar_exp.detect.pointcloud import create_points
from radar_exp.detect.cluster import cluster_points

# ----------------------------
# Widgets
# ----------------------------
cfar_method   = pn.widgets.Select(name='CFAR Method', options=['percentile', 'ca', 'axis'], value='percentile')
percentile    = pn.widgets.IntSlider(name='Percentile', start=90, end=100, value=99)
guard_cells   = pn.widgets.IntSlider(name='Guard Cells', start=0, end=12, value=2)
ref_cells     = pn.widgets.IntSlider(name='Reference Cells', start=1, end=30, value=8)
scale         = pn.widgets.FloatSlider(name='Scale', start=1.0, end=12.0, step=0.1, value=3.0)
axis_select   = pn.widgets.Select(name='Axis', options=['row', 'col'], value='row')
percentage    = pn.widgets.FloatSlider(name='Percentage', start=0.0, end=50.0, step=0.1, value=10.0)
reducer_select= pn.widgets.Select(name='Reducer', options=['median', 'mean', 'max'], value='median')
clip_min      = pn.widgets.FloatInput(name='Clip Min', value=None)
frame_idx     = pn.widgets.IntInput(name='Frame', value=0, start=0)
run_button    = pn.widgets.Button(name='Run DSP Chain', button_type='primary')
reset_zoom    = pn.widgets.Button(name='Reset Plots', button_type='default')

# hidden counter to force recompute
tick          = pn.widgets.IntInput(name='tick', value=0, visible=False)

# ----------------------------
# Load config + recording
# ----------------------------
CONFIG_PATH = Path('config/base.yaml')
cfg = yaml.safe_load(CONFIG_PATH.read_text())
dataset_cfg = cfg.get('dataset', {})
rec = RecordingAdapter(dataset_cfg.get('path'))

dsp_cfg      = cfg.get('dsp', {})
post_cfg     = cfg.get('postproc', {})

# ----------------------------
# Helper: build CFAR config from widgets
# ----------------------------
def _cfar_cfg():
    if cfar_method.value == 'percentile':
        return {'method': 'percentile', 'percentile': int(percentile.value)}
    elif cfar_method.value == 'ca':
        return {'method': 'ca',
                'guard_cells': int(guard_cells.value),
                'ref_cells':   int(ref_cells.value),
                'scale':       float(scale.value)}
    else:  # axis
        return {'method': 'axis',
                'axis': axis_select.value,
                'percentage': float(percentage.value),
                'reducer': reducer_select.value,
                'clip_min': clip_min.value}

# ----------------------------
# Core compute (pure function)
# ----------------------------
def compute_all(_tick, frame):
    """
    Returns rd_map, mask, points list, objects list for given frame and UI params.
    """
    cube = rec.get_frame(int(frame))
    rd_map, rd_cube = compute_range_doppler_map(
        cube,
        windows=dsp_cfg.get('windows', {}),
        fft_sizes=dsp_cfg.get('fft', {}),
        aggregate=dsp_cfg.get('aggregate', 'max'),
        magnitude=dsp_cfg.get('magnitude', 'db'),
    )

    coords, mask = detect_peaks(rd_map, _cfar_cfg(), post_cfg)
    points = create_points(rd_cube, rd_map, rec.max_range(), rec.max_velocity(), coords)
    objects = cluster_points(points, rd_map.shape, eps=1.0, min_samples=5)
    return rd_map, mask, points, objects

# ----------------------------
# DynamicMap producers (independent plots)
# ----------------------------
def rd_image(_tick, frame):
    rd_map, _, _, _ = compute_all(_tick, frame)
    img = hv.Image(rd_map, bounds=(0, 0, rd_map.shape[1], rd_map.shape[0]))
    return img.opts(
        cmap='inferno', colorbar=True, tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
        responsive=True, min_height=350, shared_axes=False, framewise=True, title='Range–Doppler Map'
    )

def mask_image(_tick, frame):
    rd_map, mask, _, _ = compute_all(_tick, frame)
    img = hv.Image(mask.astype(int), bounds=(0, 0, mask.shape[1], mask.shape[0]))
    return img.opts(
        cmap='viridis', colorbar=True, tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
        responsive=True, min_height=350, shared_axes=False, framewise=True, title='Detection Mask'
    )

def point_cloud(_tick, frame):
    _, _, points, objects = compute_all(_tick, frame)
    if not points:
        return hv.Points([]).opts(
            responsive=True, min_height=350, shared_axes=False, framewise=True, title='Point Cloud (Cartesian) (empty)'
        )

    xs = [p.x for p in points]  # Cartesian x
    ys = [p.y for p in points]  # Cartesian y
    base = hv.Points((xs, ys)).opts(
        size=6, color='cyan', tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
        responsive=True, min_height=350, shared_axes=False, framewise=True, title='Point Cloud (Cartesian)'
    )

    # overlay clustered objects with red "x"
    overlays = {}
    for obj in objects:
        ox = [p.x for p in obj.points]
        oy = [p.y for p in obj.points]
        overlays[obj.label] = hv.Points((ox, oy)).opts(size=9, color='red', marker='x')

    return base * hv.NdOverlay(overlays) if overlays else base

# Streamed dynamic views bound to widgets
rd_dmap   = hv.DynamicMap(pn.bind(rd_image, tick, frame_idx))
mask_dmap = hv.DynamicMap(pn.bind(mask_image, tick, frame_idx))
pc_dmap   = hv.DynamicMap(pn.bind(point_cloud, tick, frame_idx))

# ----------------------------
# Actions
# ----------------------------
def _run(_=None):
    tick.value = tick.value + 1  # triggers recompute across all DynamicMaps

def _on_param_change(_):
    tick.value = tick.value + 1  # update when any param changes

run_button.on_click(_run)
frame_idx.param.watch(_on_param_change, 'value')
cfar_method.param.watch(_on_param_change, 'value')
percentile.param.watch(_on_param_change, 'value')
guard_cells.param.watch(_on_param_change, 'value')
ref_cells.param.watch(_on_param_change, 'value')
scale.param.watch(_on_param_change, 'value')
axis_select.param.watch(_on_param_change, 'value')
percentage.param.watch(_on_param_change, 'value')
reducer_select.param.watch(_on_param_change, 'value')
clip_min.param.watch(_on_param_change, 'value')

def _reset_axes(_=None):
    # Bokeh reset tool handles this in each pane; we add a gentle nudge:
    tick.value = tick.value + 1

reset_zoom.on_click(_reset_axes)

# ---------------------------- 
# Layout (responsive, independent zoom)
# ----------------------------
controls = pn.Row(
    pn.Column(
        pn.pane.Markdown("### CFAR", sizing_mode='stretch_width'),
        cfar_method,
        pn.Row(percentile, guard_cells, ref_cells, scale, axis_select, percentage, reducer_select, clip_min),
        pn.Spacer(height=10),
        pn.Row(frame_idx, run_button, reset_zoom),
        sizing_mode='stretch_width'
    ),
)

plots = pn.Row(
    rd_dmap.opts(shared_axes=False),
    mask_dmap.opts(shared_axes=False),  
    pc_dmap.opts(shared_axes=False)
)

# Use a simpler layout instead of template
main_layout = pn.Column(
    controls,
    plots,
    sizing_mode='stretch_both'
)

# Expose serving entrypoints
def view():
    return main_layout

main_layout.servable()

if __name__ == "__main__":
    pn.serve(main_layout, show=True)
