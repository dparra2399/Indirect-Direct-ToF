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

#plt.style.use('dark_background')

breakpoint = debugger.set_trace

file = np.load('data/results/July/ntbins_2000_monte_500_exp_005.npz', allow_pickle=True)

mae = file['results']
levels_one = file['levels_one']
levels_two = file['levels_two']
params = file['params'].item()
imaging_schemes = params['imaging_schemes']
tbin_res = params['rep_tau'] / params['n_tbins']
tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)


arr = []
first_mae = mae[3, :, :]
second_mae = mae[2, :, :]
diff = first_mae - second_mae

#diff_levels = [diff.min(), -100, 0, 15, 30, 45, 60, diff.max()]
#diff_levels = np.flip(np.array([diff.max(), -40, -50, -60, -70, diff.min()]))
#colors = ['darkgreen', 'green', 'lime', 'turquoise', 'teal', 'powderblue']


plt.contourf(levels_one, levels_two, diff)#levels=diff_levels,
             #norm=mpl.colors.Normalize(vmin=diff_levels[1], vmax=diff_levels[-1], clip=True))

plt.xlabel(f' {params["levels_one"]} levels')
# plt.xticks(np.arange(0, levels_one.shape[0], step=2), np.round(levels_one[0, :][::2]))
plt.ylabel(f' {params["levels_two"]} levels')
# plt.yticks(np.arange(0, levels_two.shape[-1], step=2), np.round(levels_two[:, 0][::2]))
plt.colorbar(label="MAE Difference in MM", orientation="vertical")
plt.title(f'MAE {get_string_name(imaging_schemes[3])} - MAE {get_string_name(imaging_schemes[2])}')
plt.legend()
plt.show()


print('hello world')
