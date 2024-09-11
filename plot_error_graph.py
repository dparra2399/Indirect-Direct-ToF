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

from matplotlib import rc
import matplotlib
#matplotlib.use('TkAgg')


font = {'family': 'serif',
        'size': 9}

rc('font', **font)

rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')

#plt.style.use('dark_background')

breakpoint = debugger.set_trace

save_folder = 'Z:\\Research_Users\\David\\paper figures'
file = np.load('data\\results\\July\\ntbins_2200_monte_1000_exp_sin002.npz', allow_pickle=True)


mae = file['results']
levels_one = file['levels_one'][5:, 15:]
levels_two = file['levels_two'][5:, 15:]
params = file['params'].item()
imaging_schemes = params['imaging_schemes']
tbin_res = params['rep_tau'] / params['n_tbins']
tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)


fig = plt.figure()
ax = plt.axes(projection='3d')

arr = []

# identity = 3
# id_mae = mae[identity, :, :]
# probs = calculate_poisson_prob(levels_two, levels_one)
# fig, axs = plt.subplots(2, 1)
# axs[0].imshow(id_mae)
# axs[1].imshow(probs)
# plt.show()
for j in range(len(imaging_schemes)):
    tmp = mae[j, :, :][5:, 15:]
    #tmp[tmp > 500] = np.nan
    if imaging_schemes[j].coding_id == 'GrayTruncatedFourier':
        continue

    str_name = ''
    if imaging_schemes[j].coding_id.startswith('TruncatedFourier'):
        str_name = 'Truncated Fourier (Wide)'
    elif imaging_schemes[j].coding_id.startswith('Gated'):
        str_name = 'Coarse Hist. (Wide)'
    elif imaging_schemes[j].coding_id.startswith('Hamiltonian'):
        str_name = 'SiP Hamiltonian'
    elif imaging_schemes[j].coding_id == 'Identity':
        if imaging_schemes[j].pulse_width == 1:
            str_name = 'Full-Res. Hist. (Narrow)'
        else:
            str_name = 'Full-Res. Hist. (Wide)'
    elif imaging_schemes[j].coding_id.startswith('KTapSin'):
        if imaging_schemes[j].cw_tof is True:
            str_name = 'i-ToF Sinusoid'
        else:
            str_name = 'CoWSiP-ToF Sinusoid'
    surf = ax.plot_surface(levels_one, levels_two, tmp,
                           label=str_name, alpha=0.6,
                           edgecolors='k', lw=0.5)
    arr.append(go.Surface(z=mae[j, :, :], x=levels_one, y=levels_two))
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

#ax.view_init(elev=18., azim=50)
ax.view_init(elev=18., azim=130)
ax.set_xlabel(f' {params["levels_one"]} levels'.title())
ax.set_ylabel(f' {params["levels_two"]} levels'.title())
ax.set_zlabel('Mean Depth Error in (mm)')
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.8), fancybox=True)
#ax.set_zlim(0, 3000)
fig.tight_layout()

fig.savefig(os.path.join(save_folder, 'figure7a.svg'), bbox_inches='tight')
plt.show()

# fig = go.Figure(data=arr)
# fig.show()
print('hello world')