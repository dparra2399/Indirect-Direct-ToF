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


sigmas = [5, 10]
peak_factors = [0.030, 0.015, 0.005]

font = {'family': 'serif',
        'size': 12}

mpl.rc('font', **font)


fig, axs = plt.subplots(len(peak_factors), len(sigmas), figsize=(8, 8),squeeze=False, sharex=True, sharey=True)

diff_levels = [-3000,-150, -100, -70, -50, -30,  -10, 0, 10, 30, 50, 70, 100, 150, 3000]


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
        filename = f'ntbins_1024_monte_1000_exp_Learned_sigma{sigma}_peak{peak_name}.npz'

        file = np.load(
            f'../data/results/bandlimit_peak_simulation/{filename}',
            allow_pickle=True)

        num = 3
        num2 = 3
        mae = file['results']#[:, num2:-num, num:-num2]
        levels_one = file['levels_one']#[num2:-num, num:-num2]
        levels_two = file['levels_two']#[num:-num2, num2:-num]
        params = file['params'].item()
        imaging_schemes = params['imaging_schemes']
        tbin_res = params['rep_tau'] / params['n_tbins']
        tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)

        base = next((obj for obj in imaging_schemes if obj.coding_id == 'Identity'), None)
        target = next((obj for obj in imaging_schemes if obj.model == f'bandlimited_peak_models/n1024_k8_sigma{sigma}_peak{peak_name}_counts1000'), None)

        base_idx = imaging_schemes.index(base)
        target_idx = imaging_schemes.index(target)

        diff = mae[base_idx, :, :] - mae[target_idx, :, :]
        ##improvement_percent = ((diff) / mae[base_idx, :, :]) * 100
        im = axs[j][i].contourf(np.log10(levels_one), np.log10(levels_two), np.squeeze(diff), levels=diff_levels,
                                cmap=new_cmap,
                                norm=mpl.colors.TwoSlopeNorm(vcenter=0, vmin=diff_levels[1], vmax=diff_levels[-2]))





plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

plt.legend()
fig.subplots_adjust(wspace=0.1, hspace=0.1)
#fig.savefig(os.path.join(save_folder, 'figure5b.svg'), bbox_inches='tight')
plt.show()


print('hello world')
