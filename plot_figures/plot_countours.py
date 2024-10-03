# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from IPython.core import debugger
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
from utils.file_utils import get_string_name
import os
#plt.style.use('dark_background')

save_folder = 'Z:\\Research_Users\\David\\paper figures'

breakpoint = debugger.set_trace

file = np.load('../data/results/September/ntbins_1024_monte_1000_exp_001.npz', allow_pickle=True)

mae = file['results']
levels_one = file['levels_one']
levels_two = file['levels_two']
params = file['params'].item()
imaging_schemes = params['imaging_schemes']
tbin_res = params['rep_tau'] / params['n_tbins']
tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)


base = [3]
targets = [2]
diff_levels = [-50000, -300, -200, -150,  -100, -50, 0,  50, 75, 100, 200, 300, 50000]

font = {'family': 'serif',
        'size': 12}

mpl.rc('font', **font)

fig, axs = plt.subplots(1, len(targets), squeeze=False, sharex=True, sharey=True, figsize=(2, 3))
fig.add_subplot(111, frameon=False)
base_mae = mae[base, :, :]
original_cmap = mpl.colormaps['RdBu'].reversed()

# Define the range of the colormap you want to use (e.g., 50% of the original colormap)
start, end = 0.0, 0.5 #0.5, 1.0

# Create a new colormap that uses only a portion of the original colormap
new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'custom_cmap',
    original_cmap(np.linspace(start, end, 512))
)

for i in range(len(targets)):
    target = targets[i]
    diff = base_mae - mae[target, :, :]
    improvement_percent = ((base_mae - mae[target, :, :]) / base_mae) * 100
    im = axs[0][i].contourf(levels_one, levels_two, np.squeeze(improvement_percent), levels=diff_levels,
                            cmap=original_cmap, norm=mpl.colors.Normalize(vmin=diff_levels[1], vmax=diff_levels[-2], clip=True))
    str_name = ''
    if imaging_schemes[target].coding_id.startswith('TruncatedFourier'):
        str_name = 'Truncated Fourier \n (Wide)'
        axs[0][i].set_title(str_name)
    elif imaging_schemes[target].coding_id.startswith('Gated'):
        str_name = 'Coarse Hist. \n (Wide)'
        axs[0][i].set_title(str_name)
    elif imaging_schemes[target].coding_id.startswith('Hamiltonian'):
        str_name = 'SiP Hamiltonian'
        axs[0][i].set_title(str_name, color='red')
    elif imaging_schemes[target].coding_id == 'Identity':
        if imaging_schemes[target].pulse_width == 1:
            str_name = 'Full-Res. Hist. \n (Narrow)'
            axs[0][i].set_title(str_name)
        else:
            str_name = 'Full-Res. Hist. \n (Wide)'
            axs[0][i].set_title(str_name)


plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
#plt.ylabel(f'Average Ambient Photon per Bin', labelpad=10)
#plt.xlabel(f'Peak Photon Count')


cbar_ticks = diff_levels[1:-1]
cbar = fig.colorbar(im, ticks=cbar_ticks, orientation='horizontal', ax=axs)

plt.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_folder, 'figure5c.svg'), bbox_inches='tight')
plt.subplots_adjust(hspace=0.5, wspace=0.05)
plt.show()


print('hello world')
