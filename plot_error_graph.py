# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from IPython.core import debugger
from felipe_utils.felipe_impulse_utils import tof_utils_felipe

breakpoint = debugger.set_trace

file = np.load('data/results/ntbins_976_monte_1000.npz', allow_pickle=True)


mae = file['results']
levels_one = file['levels_one']
levels_two = file['levels_two']
params = file['params'].item()
imaging_schemes = params['imaging_schemes']
tbin_res = params['rep_tau'] / params['n_tbins']
tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)


fig = plt.figure()
ax = plt.axes(projection='3d')

for j in range(len(imaging_schemes)):
    surf = ax.plot_surface(np.log10(levels_one), np.log10(levels_two), mae[j, :, :],
                           label=imaging_schemes[j].coding_id)
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

ax.set_xlabel(f'log10 {params["levels_one"]} levels')
ax.set_ylabel(f'log10 {params["levels_two"]} levels')
ax.set_zlabel('mean absolute error in (mm)')
ax.legend()
plt.show()

print('hello world')