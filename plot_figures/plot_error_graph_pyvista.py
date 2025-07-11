import numpy as np
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plot_figures.plot_utils import *

fig = go.Figure()

save_folder = '/Volumes/velten/Research_Users/David/ICCP 2025 Hardware-aware codes/Learned Coding Functions Paper'
# filenames = [
#             '../data/results/bandlimit_simulation/ntbins_1024_monte_5000_exp_Learned_sigma30_mae.npz',
#             '../data/results/bandlimit_simulation/ntbins_1024_monte_5000_exp_Learned_sigma30_rmse.npz',
#
# ]
# #
# filenames = [
# #             '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma1_peak030_rmse.npz',
#
#             '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma10_peak015_mae.npz',
#             '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma5_peak015_mae.npz',
#             # '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma10_peak015_mae.npz',
#
#     #'../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma10_peak015_rmse.npz',
#
# ]
filenames = [
             '../data/results/peak_simulation_constant_pulse_energy/ntbins_1024_monte_5000_exp_Learned_n1024_k14_peak015_mae_constant_pulse_energy.npz',
             '../data/results/peak_simulation_constant_pulse_energy/ntbins_1024_monte_5000_exp_Learned_n1024_k14_peak015_rmse_constant_pulse_energy.npz',
            #'../data/results/peak_simulation_constant_pulse_energy/ntbins_1024_monte_5000_exp_Learned_peak015_mae_constant_pulse_energy.npz',

    #             '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_2000_exp_Learned_sigma10_peak015_rmse_constant_pulse_energy.npz',

]
# filenames = [
#      '../data/results/bandlimit_simulation/ntbins_2000_monte_5000_exp_Learned_n2000_k10_sigma30_mae.npz',
#      '../data/results/bandlimit_simulation/ntbins_2000_monte_5000_exp_Learned_n2000_k10_sigma30_rmse.npz',
#
#      #'../data/results/bandlimit_simulation/ntbins_2000_monte_5000_exp_Learned_n2000_k12_sigma30_mae.npz',
#      #'../data/results/bandlimit_simulation/ntbins_2000_monte_5000_exp_Learned_n2000_k14_sigma30_mae.npz',
#
#  ]
num = 8 #high SBR
num2 = 11 #Low Photon count
num3 = 1 #Low SBR
num4 = 1 #High photon count
grid_size = 7

n_files = len(filenames)
n_cols = 2
n_rows = math.ceil(n_files / n_cols)

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    specs=[[{'type': 'surface'}]*n_cols]*n_rows,
    horizontal_spacing=0.05, vertical_spacing=0.1,
    subplot_titles=[f.split('/')[-1] for f in filenames]
)


for idx, filename in enumerate(filenames):
    file = np.load(filename, allow_pickle=True)
    mae = file['results'][:, num3:-num, num2:-num4] * (1/10)
    levels_one = file['levels_one'][num3:-num, num2:-num4]
    levels_two = file['levels_two'][num3:-num, num2:-num4]
    imaging_schemes = file['params'].item()['imaging_schemes']

    row = idx // n_cols + 1
    col = idx % n_cols + 1

    for j, scheme in enumerate(imaging_schemes):
        tmp = mae[j, :, :]
        label = ''

        # if idx == 0 and (j == 7 or j == 8 or j == 6):
        #     continue
        # if idx == 1 and (j == 6 or j == 8 or j == 7):
        #     continue
        # if idx == 2 and (j == 7 or j == 6):
        #     continue
        # if idx == 0 and (j == 1 or j == 2):
        #     continue
        # if idx == 1 and (j == 1 or j == 3):
        #     continue
        # if idx == 2 and (j == 1 or j == 2):
        #     continue

        if scheme.coding_id == 'Greys':
            continue
        hex_color = get_scheme_color(
            imaging_schemes[j].coding_id, 8,
            cw_tof=imaging_schemes[j].cw_tof,
            constant_pulse_energy=False
        )
        if len(hex_color) == 9 and hex_color.startswith('#'):
            hex_color = hex_color[:7]

        fig.add_trace(go.Surface(
            z=tmp,
            x=np.log10(levels_one),
            y=np.log10(levels_two),
            surfacecolor=np.ones_like(tmp),
            colorscale=[[0, hex_color], [1, hex_color]],
            cmin=0,
            cmax=1,
            name=scheme.coding_id,
            showscale=False,
            contours=dict(
                x=dict(show=True, color='#4d4d4d', width=2),
                y=dict(show=True, color='#4d4d4d', width=2),

            )
        ), row=row, col=col)
# Layout
xticks = np.round(np.linspace(np.min(np.log10(levels_one)), np.max(np.log10(levels_one)), num=grid_size), 2)
yticks = np.round(np.linspace(np.min(np.log10(levels_two)), np.max(np.log10(levels_two)), num=grid_size), 2)

n = len(filenames)
scene_layouts = {}

gap = 0.015 # spacing between plots
width = (1 - (n - 1) * gap) / n  # dynamic width per subplot based on gap and count

for i in range(1, n + 1):
    scene_key = f"scene{i}" if i > 1 else "scene"
    start = (i - 1) * (width + gap)
    end = start + width
    print(filenames[i-1].split("."))

    if filenames[i-1].split(".")[-2].split("_")[-4].upper() == 'MAE':
        z_end = 30
    else:
        z_end = 50
    scene_layouts[scene_key] = dict(
        domain=dict(x=[start, end]),
        xaxis=dict(
            title=dict(text='<br>Log Photon Count', font=dict(family='serif', size=25, color='black')),
            tickmode='array',
            tickvals=xticks,
            ticktext=[f'{v:.1f}' for v in xticks],
            tickfont=dict(family='serif', size=15, color='black'),
            showgrid=True,
            gridcolor='lightgray',
            backgroundcolor='white',
        ),
        yaxis=dict(
            title=dict(text='Log SBR', font=dict(family='serif', size=25, color='black')),
            tickmode='array',
            tickvals=yticks,
            ticktext=[f'{v:.1f}' for v in yticks],
            tickfont=dict(family='serif', size=15, color='black'),
            showgrid=True,
            gridcolor='lightgray',
            backgroundcolor='white',
        ),
        zaxis=dict(
            title=dict(text=f'{filenames[i-1].split(".")[-2].split("_")[-4].upper()} (cm)', font=dict(family='serif', size=25, color='black')),
            tickfont=dict(family='serif', size=15, color='black'),
            showgrid=True,
            gridcolor='lightgray',
            backgroundcolor='white',
            range=[0.1, z_end]
        ),
        camera=dict(eye=dict(x=2.0, y=-2.0, z=0.7)),
        bgcolor='white'
    )

fig.update_layout(
    **scene_layouts,
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=10, r=10, t=40, b=10),
    width=600 * n,
    height=700,
)



fig.write_image("plot_output.svg")

import plotly.io as pio
pio.renderers.default = 'browser'
fig.show()