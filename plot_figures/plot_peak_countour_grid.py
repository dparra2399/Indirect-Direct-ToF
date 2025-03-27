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
grid_size = 3
metric = 'mae'

font = {'family': 'serif',
        'size': 12}

mpl.rc('font', **font)


fig, axs = plt.subplots(len(peak_factors), len(sigmas), figsize=(9, 9),squeeze=False, sharex=True, sharey=True)

diff_levels = [-2000,-20, -10, -7.5,  -5, -3,  -1, 0, 1, 3, 5, 7.5, 10, 20, 50]


#fig.add_subplot(111, frameon=False)
original_cmap = mpl.colormaps['RdBu'].reversed()

# Define the range of the colormap you want to use (e.g., 50% of the original colormap)
start, end = 0.0, 1.0 #0.5, 1.0

# Create a new colormap that uses only a portion of the original colormap
new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'custom_cmap',
    original_cmap(np.linspace(start, end, 512))
)

#new_cmap = darken_cmap(new_cmap, factor=0.8)


for i in range(len(sigmas)):
    sigma = sigmas[i]

    for j in range(len(peak_factors)):
        peak_factor = peak_factors[j]
        peak_name =  f"{peak_factor:.3f}".split(".")[-1]
        filename = f'ntbins_1024_monte_1000_exp_Learned_sigma{sigma}_peak{peak_name}_{metric}.npz'

        file = np.load(
            f'../data/results/bandlimit_peak_simulation/{filename}',
            allow_pickle=True)

        num = 3
        num2 = 3


        axs[-1][j].set_xlabel('Log Photon Counts')
        axs[j][0].set_ylabel('Log SBR')

        mae = file['results'][:, num2:-num, num:-num2]
        levels_one = file['levels_one'][num2:-num, num:-num2]
        levels_two = file['levels_two'][num:-num2, num2:-num]
        params = file['params'].item()
        imaging_schemes = params['imaging_schemes']
        tbin_res = params['rep_tau'] / params['n_tbins']
        tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)

        target = next((obj for obj in imaging_schemes if obj.coding_id == 'TruncatedFourier'), None)
        base = next((obj for obj in imaging_schemes if obj.model == f'bandlimited_peak_models/n1024_k8_sigma{sigma}_peak{peak_name}_counts1000'), None)

        base_idx = imaging_schemes.index(base)
        target_idx = imaging_schemes.index(target)

        diff = (mae[base_idx, :, :] - mae[target_idx, :, :]) * (1/10)
        ##improvement_percent = ((diff) / mae[base_idx, :, :]) * 100
        im = axs[j][i].contourf(np.log10(levels_one), np.log10(levels_two), np.squeeze(diff), levels=diff_levels,
                                cmap=new_cmap,
                                norm=mpl.colors.TwoSlopeNorm(vcenter=0, vmin=diff_levels[1], vmax=diff_levels[-1]))

        # axs[j][i].set_xticks(np.round(np.linspace(np.min(np.log10(levels_one)), np.max(np.log10(levels_one)), num=grid_size),
        #                        1))  # Set x-axis ticks
        # axs[j][i].set_yticks(np.round(np.linspace(np.min(np.log10(levels_two)), np.max(np.log10(levels_two)), num=grid_size),
        #                        1))  # Set y-axis ticks

        # Optionally, customize tick labels
        # axs[j][i].set_xticklabels(
        #     np.round(np.linspace(np.min(np.log10(levels_one)), np.max(np.log10(levels_one)), num=grid_size), 1),
        #     fontsize=12)
        # axs[j][i].set_yticklabels(
        #     np.round(np.linspace(np.min(np.log10(levels_two)), np.max(np.log10(levels_two)), num=grid_size), 1),
        #     fontsize=12)





plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
cbar_ax = fig.add_axes([0.05, -0.0001, 0.95, 0.05]) # [left, bottom, width, height]
cbar_ticks = diff_levels[1:]
print(diff_levels)
print(cbar_ticks)

cbar_labels = [f'>{abs(diff_levels[0])} \n cm']
cbar_labels = ['> 2k \n cm']
cbar_labels.append('2k - \n 20 cm')
#cbar_labels = []
cbar_labels.extend([f'{abs(diff_levels[i-1])} -\n {abs(diff_levels[i])} cm' for i in range(2, len(cbar_ticks[:-1]) // 2)])
cbar_labels.extend([f'{abs(diff_levels[i])} -\n {abs(diff_levels[i-1])} cm' for i in range(len(cbar_ticks[:-1]) // 2, len(cbar_ticks))])
#cbar_labels.append(f'> {cbar_ticks[-1]} \n cm')

print(cbar_labels)

cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ax=axs.flatten(), ticks=cbar_ticks)
cbar.set_ticklabels(cbar_labels)

cbar.ax.tick_params(axis='x', direction='in', length=32, pad=-26)  # Adjust length and paddin
for t in cbar.ax.xaxis.get_ticklabels():
    t.set_horizontalalignment('right')
    t.set_x(t.get_position()[0] - 0.2)  # Shift them slightly left# Move labels to the left
cbar.ax.tick_params(axis='y', pad=-100)

plt.legend()
fig.subplots_adjust(wspace=0.1, hspace=0.1)
#fig.savefig(r'Z:\Research_Users\David\Learned Coding Functions Paper\contour_grid.svg', bbox_inches='tight')
plt.show()


print('hello world')
