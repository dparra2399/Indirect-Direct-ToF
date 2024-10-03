# Python imports
# Library imports
import plotly.graph_objects as go
from IPython.core import debugger
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
from plot_figures.plot_utils import *

from matplotlib import rc

matplotlib.use('TkAgg')


font = {'family': 'serif',
        'size': 12}

rc('font', **font)

rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')

#plt.style.use('dark_background')

breakpoint = debugger.set_trace

save_folder = 'Z:\\Research_Users\\David\\paper figures'
file = np.load('../data/results/September/ntbins_1024_monte_1000_exp_Gated002.npz', allow_pickle=True)

mae = file['results'][:, 3:-3, 3:-3]
levels_one = file['levels_one'][3:-3, 3:-3]
levels_two = file['levels_two'][3:-3, 3:-3]
params = file['params'].item()
imaging_schemes = params['imaging_schemes']
tbin_res = params['rep_tau'] / params['n_tbins']
tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)


fig = plt.figure()
ax = plt.axes(projection='3d')

arr = []

for j in range(len(imaging_schemes)):
    tmp = mae[j, :, :]
    #tmp[tmp > 500] = np.nan
    # if imaging_schemes[j].coding_id == 'GrayTruncatedFourier':
    #     continue

    str_name = ''
    if imaging_schemes[j].coding_id.startswith('TruncatedFourier'):
        str_name = 'Truncated Fourier (Wide)'

    elif imaging_schemes[j].coding_id.startswith('Gated'):
        str_name = 'Coarse Hist. (Wide)'
    elif imaging_schemes[j].coding_id.startswith('Hamiltonian'):
        str_name = f'SiP Hamiltonian K={imaging_schemes[j].coding_id[-1]}'
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

    elif imaging_schemes[j].coding_id == 'Greys':
        str_name = 'Count. Greys'


    surf = ax.plot_surface(levels_one, levels_two, tmp,
                           label=str_name, alpha=0.6,
                           edgecolors='k', lw=0.5,
                           color=get_scheme_color(imaging_schemes[j].coding_id, pw=imaging_schemes[j].pulse_width))
    arr.append(go.Surface(z=mae[j, :, :], x=levels_one, y=levels_two))
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

ax.view_init(elev=15., azim=35)
#ax.view_init(elev=18., azim=130)
#plt.ylabel(f'Average Ambient Photon per Bin', labelpad=10)
#plt.xlabel(f'Peak Photon Count')

#ax.set_zlabel('Mean Depth Error in (mm)')
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.8), fancybox=True)
ax.set_zlim(0, 500)
fig.tight_layout()

fig.savefig(os.path.join(save_folder, 'figure6b.svg'), bbox_inches='tight', dpi=3000)
plt.show()

# fig = go.Figure(data=arr)
# fig.show()
print('hello world')