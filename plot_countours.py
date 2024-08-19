# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from IPython.core import debugger
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
from utils.file_utils import get_string_name
import os
#plt.style.use('dark_background')

save_folder = '/Users/Patron/Desktop/cowsip figures'

breakpoint = debugger.set_trace

file = np.load('data/results/August/ntbins_1024_monte_1000_exp_001_August8.npz', allow_pickle=True)

mae = file['results']
levels_one = file['levels_one']
levels_two = file['levels_two']
params = file['params'].item()
imaging_schemes = params['imaging_schemes']
tbin_res = params['rep_tau'] / params['n_tbins']
tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)


base = [4]
targets = [[3]]
diff_levels = [-10000, -100, -90, -70, -50, -30, -10, 0, 10, 30, 50, 70, 90, 100, 1000]

font = {'family': 'normal',
        'size': 10}

mpl.rc('font', **font)

fig, axs = plt.subplots(len(targets), len(targets[0]), squeeze=False, sharex=True, sharey=True, figsize=(7, 7))
fig.add_subplot(111, frameon=False)
base_mae = mae[base, :, :]
for i in range(len(targets)):
    for j in range(len(targets[0])):
        target = targets[i][j]
        diff = base_mae - mae[target, :, :]
        im = axs[i][j].contourf(levels_one, levels_two, np.squeeze(diff), levels=diff_levels,
                               cmap='bwr', norm=mpl.colors.Normalize(vmin=diff_levels[1], vmax=diff_levels[-2], clip=True))
        axs[i][j].set_title(f'{get_string_name(imaging_schemes[target])}')
        if i == 1:
            axs[i][j].set_xlabel('Max Photon per Bin $P_{max}$')

plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel(f' {params["levels_one"]} levels')
plt.ylabel(f'Average Ambient Photon per Bin', labelpad=10)
cbar_ticks = diff_levels[1:-1]
cbar = fig.colorbar(im, ticks=cbar_ticks, orientation='horizontal', ax=axs)
plt.legend()

fig.savefig(os.path.join(save_folder, 'figure4.jpg'), bbox_inches='tight')

plt.show()


print('hello world')
