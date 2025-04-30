# Python imports
# Library imports
import numpy as np
import matplotlib as mpl


mpl.use('TkAgg')
import matplotlib.pyplot as plt
from IPython.core import debugger
from felipe_utils import tof_utils_felipe
import os
from plot_utils import darken_cmap
#plt.style.use('dark_background')

breakpoint = debugger.set_trace


sigmas = [10, 20, 30]
targets = ['TruncatedFourier', 'Greys']
grid_size = 4
metric = 'rmse'

font = {'family': 'serif',
        'size': 12}

mpl.rc('font', **font)


fig, axs = plt.subplots(len(targets), len(sigmas), figsize=(9, 6),squeeze=False, sharex=True, sharey=True)

diff_levels = [-10, -5, -3,  -2, -1,  -0.5, -0.3, -0.2, -0.1, 0, 1, 5, 10, 50]#1, 2, 3, 4, 5, 6, 100]


#fig.add_subplot(111, frameon=False)
original_cmap = mpl.colormaps['RdBu'].reversed()

# Define the range of the colormap you want to use (e.g., 50% of the original colormap)
start, end = 0, 1.0 #0.5, 1.0

# Create a new colormap that uses only a portion of the original colormap
new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'custom_cmap',
    original_cmap(np.linspace(start, end, 512)) ** 3
)

#new_cmap = darken_cmap(new_cmap, factor=0.9)


for i in range(len(sigmas)):
    sigma = sigmas[i]

    for j in range(len(targets)):
        target = targets[j]
        filename = f'ntbins_1024_monte_5000_exp_Learned_sigma{sigma}_{metric}.npz'

        file = np.load(
            f'../data/results/bandlimit_simulation/{filename}',
            allow_pickle=True)

        num = 8
        num2 = 2

        mae = file['results'][:, :-num, num2:]
        levels_one = file['levels_one'][:-num, num2:]  # [num2:-num, num2:-num]
        print(np.min(levels_one))
        levels_two = file['levels_two'][:-num, num2:]  # [num2:-num, num2:-num]
        params = file['params'].item()
        imaging_schemes = params['imaging_schemes']
        tbin_res = params['rep_tau'] / params['n_tbins']
        tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)

        target = next((obj for obj in imaging_schemes if obj.coding_id == target), None)
        base = next((obj for obj in imaging_schemes if obj.model == f'bandlimited_models/n1024_k8_sigma{sigma}'), None)

        base_idx = imaging_schemes.index(base)
        target_idx = imaging_schemes.index(target)

        diff = np.clip((mae[base_idx, :, :] - mae[target_idx, :, :]) * (1/10), diff_levels[0],diff_levels[-1])

        ##improvement_percent = ((diff) / mae[base_idx, :, :]) * 100
        im = axs[j][i].contourf(np.log10(levels_one), np.log10(levels_two), np.squeeze(diff), levels=diff_levels,
                                cmap=new_cmap,
                                norm=mpl.colors.TwoSlopeNorm(vcenter=0, vmin=diff_levels[0], vmax=diff_levels[-1]))

        CS2 = axs[j][i].contour(im, levels=im.levels[1:, ...], colors='skyblue', linewidths=0.5, linestyles='-')

        axs[j][i].set_xticks(np.round(np.linspace(np.min(np.log10(levels_one))+.1, np.max(np.log10(levels_one))-.1, num=grid_size),
                               1))  # Set x-axis ticks
        axs[j][i].set_yticks(np.round(np.linspace(np.min(np.log10(levels_two))+.1, np.max(np.log10(levels_two))-.1, num=grid_size),
                               1))  # Set y-axis ticks

        axs[j][i].set_position([0.1, 0.1, 0.8, 0.8])




axs[-1][1].set_xlabel('Log Photon Counts')
axs[1][0].set_ylabel('Log SBR')
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
cbar_ax = fig.add_axes([0.05, -0.0001, 0.95, 0.06]) # [left, bottom, width, height]
cbar_ticks = diff_levels[1:]
print(diff_levels)
print(cbar_ticks)

cbar_labels = [f'>{abs(diff_levels[1])}']
#cbar_labels = ['> ']
#cbar_labels.append(f'{abs(diff_levels[0])}-{abs(diff_levels[1])}')
cbar_labels.extend([f'{min(abs(diff_levels[i]), abs(diff_levels[i-1]))}-{max(abs(diff_levels[i]), abs(diff_levels[i-1]))}' for i in range(2, len(cbar_ticks))])
cbar_labels.append(f'> {cbar_ticks[-2]} \n')

print(cbar_labels)

cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ax=axs.flatten(), ticks=cbar_ticks)
cbar.set_ticklabels(cbar_labels)

cbar.ax.tick_params(axis='x', direction='in', length=0, pad=-15)  # Adjust length and paddin
for t in cbar.ax.xaxis.get_ticklabels():
    t.set_horizontalalignment('right')
    #t.set_verticalalignment('center')

    t.set_x(t.get_position()[0] - 0.2)  # Shift them slightly left# Move labels to the left
#cbar.ax.tick_params(axis='y', pad=-100)
cbar.add_lines(CS2)

plt.legend()
fig.subplots_adjust(wspace=0.04, hspace=0.04)
fig.savefig(r'Z:\Research_Users\David\Learned Coding Functions Paper\contour_band_grid.svg', bbox_inches='tight')
plt.show()


print('hello world')
