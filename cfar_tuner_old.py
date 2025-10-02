"""
Interactive CFAR tuner using Holoviews and Bokeh.

Allows tuning of CFAR parameters, runs the DSP chain, and visualizes RD map, detection mask, and point cloud/object detections.
"""
import holoviews as hv
import panel as pn
import numpy as np
import yaml
hv.extension('bokeh')

from radar_exp.io.recording_adapter import RecordingAdapter
from radar_exp.dsp.rd_map import compute_range_doppler_map
from radar_exp.detect.peaks import detect_peaks
from radar_exp.detect.pointcloud import create_points
from radar_exp.detect.cluster import cluster_points

# --- UI widgets ---
cfar_method = pn.widgets.Select(name='CFAR Method', options=['percentile', 'ca'])
percentile = pn.widgets.IntSlider(name='Percentile', start=90, end=100, value=99)
guard_cells = pn.widgets.IntSlider(name='Guard Cells', start=0, end=10, value=2)
ref_cells = pn.widgets.IntSlider(name='Reference Cells', start=1, end=20, value=8)
scale = pn.widgets.FloatSlider(name='Scale', start=1.0, end=10.0, value=3.0)
frame_idx = pn.widgets.IntInput(name='Frame Index', value=0, start=0)
run_button = pn.widgets.Button(name='Run DSP Chain', button_type='primary')

# --- Data loading ---
config_path = 'config/base.yaml'  # You can change this path
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
dataset_cfg = config.get('dataset', {})
rec_path = dataset_cfg.get('path')
rec = RecordingAdapter(rec_path)

# --- DSP chain and visualization ---
def run_chain(event=None):
    cube = rec.get_frame(frame_idx.value)
    dsp_cfg = config.get('dsp', {})
    windows_cfg = dsp_cfg.get('windows', {})
    fft_sizes = dsp_cfg.get('fft', {})
    aggregate = dsp_cfg.get('aggregate', 'max')
    magnitude = dsp_cfg.get('magnitude', 'db')
    rd_map, rd_cube = compute_range_doppler_map(
        cube,
        windows=windows_cfg,
        fft_sizes=fft_sizes,
        aggregate=aggregate,
        magnitude=magnitude,
    )
    # CFAR config
    if cfar_method.value == 'percentile':
        cfar_cfg = {'method': 'percentile', 'percentile': percentile.value}
    else:
        cfar_cfg = {'method': 'ca', 'guard_cells': guard_cells.value, 'ref_cells': ref_cells.value, 'scale': scale.value}
    post_cfg = config.get('postproc', {})
    coords, mask = detect_peaks(rd_map, cfar_cfg, post_cfg)
    points = create_points(rd_cube, rd_map, rec.max_range(), rec.max_velocity(), coords)
    objects = cluster_points(points, rd_map.shape, eps=1.0, min_samples=5)
    # --- Visualizations ---
    # RD map
    rd_img = hv.Image(rd_map, bounds=(0,0,rd_map.shape[1],rd_map.shape[0])).opts(cmap='inferno', colorbar=True, title='RD Map')
    # Detection mask
    mask_img = hv.Image(mask.astype(int), bounds=(0,0,mask.shape[1],mask.shape[0])).opts(cmap='viridis', colorbar=True, title='Detection Mask')
    # Point cloud
    if points:
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        pc_scatter = hv.Points((xs, ys)).opts(size=8, color='cyan', title='Point Cloud', tools=['hover'])
    else:
        pc_scatter = hv.Points([])
    # Object detections
    obj_scatter = hv.NdOverlay({obj.label: hv.Points(([p.x for p in obj.points], [p.y for p in obj.points])).opts(size=10, color='red', marker='x', title=f'Object {obj.label}') for obj in objects})
    # Layout
    layout = (rd_img + mask_img + pc_scatter * obj_scatter).cols(2)
    return layout

run_button.on_click(run_chain)

# --- Panel layout ---
cfar_panel = pn.Column(
    pn.Row(cfar_method, percentile, guard_cells, ref_cells, scale),
    frame_idx,
    run_button,
    pn.bind(run_chain)
)

cfar_panel.servable()

if __name__ == '__main__':
    pn.serve(cfar_panel, show=True)
