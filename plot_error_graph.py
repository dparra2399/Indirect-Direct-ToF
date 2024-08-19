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
from utils.plot_utils import *
#plt.style.use('dark_background')

breakpoint = debugger.set_trace

file = np.load('data/results/August/ntbins_1024_monte_1000_exp_001_August8.npz', allow_pickle=True)


mae = file['results']
levels_one = file['levels_one'][:, :]
levels_two = file['levels_two'][:, :]
params = file['params'].item()
imaging_schemes = params['imaging_schemes']
tbin_res = params['rep_tau'] / params['n_tbins']
tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
#

arr = []

identity = 3
id_mae = mae[identity, :, :]
probs = calculate_poisson_prob(levels_two, levels_one)
fig, axs = plt.subplots(2, 1)
axs[0].imshow(id_mae)
axs[1].imshow(probs)
plt.show()
for j in range(len(imaging_schemes)):
    tmp = mae[j, :, :][:, :]
    tmp[tmp > 500] = np.nan
    if imaging_schemes[j].coding_id == 'GrayTruncatedFourier':
        continue
    surf = ax.plot_surface(levels_one, np.log10(levels_two), tmp,
                           label=get_string_name(imaging_schemes[j]), alpha=0.6,
                           edgecolors='k', lw=0.6)
    arr.append(go.Surface(z=mae[j, :, :], x=levels_one, y=np.log10(levels_two)))
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

ax.set_xlabel(f' {params["levels_one"]} levels')
ax.set_ylabel(f' {params["levels_two"]} levels')
ax.set_zlabel('mean absolute error in (mm)')
ax.legend()
ax.set_zlim(0, 500)
plt.show()

# fig = go.Figure(data=arr)
# fig.show()
print('hello world')