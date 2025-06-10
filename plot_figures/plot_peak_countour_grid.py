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


sigmas = [1, 5, 10]
peak_factors = [0.030, 0.015, 0.005]
grid_size = 4
metric = 'rmse'

font = {'family': 'serif',
        'size': 12}

mpl.rc('font', **font)


fig, axs = plt.subplots(len(peak_factors), len(sigmas), figsize=(9, 9),squeeze=False, sharey=True)

diff_levels = [-100,-50, -25, -10, -5, -1, 0, 1, 10, 50]


#fig.add_subplot(111, frameon=False)
original_cmap = mpl.colormaps['RdBu'].reversed()

# Define the range of the colormap you want to use (e.g., 50% of the original colormap)
start, end = 0.0, 1.0 #0.5, 1.0

# Create a new colormap that uses only a portion of the original colormap
new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'custom_cmap',
    original_cmap(np.linspace(start, end, 512)) ** 3
)

#new_cmap = darken_cmap(new_cmap, factor=0.9)


for i in range(len(sigmas)):
    sigma = sigmas[i]

    for j in range(len(peak_factors)):
        peak_factor = peak_factors[j]
        peak_name =  f"{peak_factor:.3f}".split(".")[-1]
        filename = f'ntbins_1024_monte_5000_exp_Learned_sigma{sigma}_peak{peak_name}_{metric}.npz'

        file = np.load(
            f'../data/results/bandlimit_peak_simulation/{filename}',
            allow_pickle=True)

        num = 8  # high SBR
        num2 = 2  # Low Photon count
        num3 = 1  # Low SBR
        num4 = 1  # High photon count
        grid_size = 4

        mae = file['results'][:, num3:-num, num2:-num4] * (1 / 10)  # [:, num2:-num, num2:-num] * (1/10)
        levels_one = file['levels_one'][num3:-num, num2:-num4]  # [num2:-num, num2:-num]
        print(np.min(levels_one))
        levels_two = file['levels_two'][num3:-num, num2:-num4]  # [num2:-num, num2:-num]
        params = file['params'].item()
        imaging_schemes = params['imaging_schemes']
        tbin_res = params['rep_tau'] / params['n_tbins']
        tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)

        target = next((obj for obj in imaging_schemes if obj.coding_id == 'Identity'), None)
        base = next((obj for obj in imaging_schemes if obj.model == f'bandlimited_peak_models/n1024_k8_sigma{sigma}_peak{peak_name}_counts1000'), None)

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


        #axs[j][i].set_aspect('equal')
        axs[j][i].set_position([0.1, 0.1, 0.8, 0.8])


axs[-1][1].set_xlabel('Log Photon Counts')
axs[1][0].set_ylabel('Log SBR')
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
cbar_ax = fig.add_axes([0.05, -0.0001, 0.95, 0.05]) # [left, bottom, width, height]
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

cbar.ax.tick_params(axis='x', direction='in', length=0, pad=-26)  # Adjust length and paddin
for t in cbar.ax.xaxis.get_ticklabels():
    t.set_horizontalalignment('right')
    #t.set_verticalalignment('center')

    t.set_x(t.get_position()[0] - 0.2)  # Shift them slightly left# Move labels to the left
#cbar.ax.tick_params(axis='y', pad=-100)
cbar.add_lines(CS2)

plt.legend()
fig.subplots_adjust(wspace=0.04, hspace=0.04)
fig.savefig(r'/Volumes/velten/Research_Users/David/ICCP 2025 Hardware-aware codes/Learned Coding Functions Paper/tmp2.svg', bbox_inches='tight')
plt.show()


print('hello world')
