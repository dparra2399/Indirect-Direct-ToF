# Python imports
# Library imports
import numpy as np
from IPython.core import debugger
from felipe_utils import tof_utils_felipe
from plot_figures.plot_utils import *
from utils.file_utils import get_string_name

from matplotlib import rc


font = {'family': 'serif',
        'size': 10}

rc('font', **font)


#plt.style.use('dark_background')

breakpoint = debugger.set_trace

save_folder = '/Volumes/velten/Research_Users/David/ICCP 2025 Hardware-aware codes/Learned Coding Functions Paper'
filenames = [
            '../data/results/bandlimit_simulation/ntbins_2000_monte_5000_exp_Learned_n2000_k10_sigma30_mae.npz',
            '../data/results/bandlimit_simulation/ntbins_2000_monte_5000_exp_Learned_n2000_k12_sigma30_mae.npz',
            '../data/results/bandlimit_simulation/ntbins_2000_monte_5000_exp_Learned_n2000_k14_sigma30_mae.npz',

    # '../data/results/bandlimit_simulation/ntbins_1024_monte_5000_exp_Learned_sigma10_mae.npz',
            #'../data/results/bandlimit_simulation/ntbins_1024_monte_5000_exp_Learned_sigma10_rmse.npz'
            #'../data/results/bandlimit_simulation/ntbins_1024_monte_2000_exp_Learned_sigma10_mae_photonstarved.npz',
            #'../data/results/bandlimit_simulation/ntbins_1024_monte_2000_exp_Learned_sigma10_rmse_photonstarved.npz'

            ]
# filenames = [
# #             '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma1_peak030_rmse.npz',
#
#             '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma5_peak005_mae.npz',
#             '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma5_peak005_rmse.npz',
#             # '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma10_peak015_mae.npz',
#
#     #'../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma10_peak015_rmse.npz',
#
# ]

# filenames = [
#              '../data/results/peak_simulation_constant_pulse_energy/ntbins_1024_monte_5000_exp_Learned_peak005_mae_constant_pulse_energy.npz',
#             #'../data/results/peak_simulation_constant_pulse_energy/ntbins_1024_monte_5000_exp_Learned_peak030_mae_constant_pulse_energy.npz',
#             '../data/results/peak_simulation_constant_pulse_energy/ntbins_1024_monte_5000_exp_Learned_peak005_rmse_constant_pulse_energy.npz',
#
#     #             '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_2000_exp_Learned_sigma10_peak015_rmse_constant_pulse_energy.npz',
#
# ]

fig, axs = plt.subplots(1, len(filenames), subplot_kw={"projection": "3d"}, figsize=(15, 10), squeeze=False)

num = 8 #high SBR
num2 = 3 #Low Photon count
num3 = 1 #Low SBR
num4 = 1 #High photon count
grid_size = 4

for i, filename in enumerate(filenames):
    file = np.load(filename, allow_pickle=True)

    mae = file['results'][:, num3:-num, num2:-num4] * (1/10) #[:, num2:-num, num2:-num] * (1/10)
    levels_one = file['levels_one'][num3:-num, num2:-num4]#[num2:-num, num2:-num]
    print(np.min(levels_one))
    levels_two = file['levels_two'][num3:-num, num2:-num4]#[num2:-num, num2:-num]
    params = file['params'].item()
    imaging_schemes = params['imaging_schemes']
    tbin_res = params['rep_tau'] / params['n_tbins']
    tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)


    for j in range(len(imaging_schemes)):
        tmp = mae[j, :, :]
        #tmp[tmp > 500] = np.nan
        # if imaging_schemes[j].coding_id == 'GrayTruncatedFourier':
        #     continue

        str_name = ''
        # if i == 0 and (j == 7 or j == 8):
        #     continue
        # if i == 1 and (j == 6 or j == 8):
        #     continue
        # if i == 2 and (j == 7 or j == 6):
        #     continue
        # if i == 0 and (j == 2 or j == 3):
        #     continue
        # if i == 1 and (j == 1 or j == 3):
        #     continue
        # if i == 2 and (j == 1 or j == 2):
        #     continue
        if imaging_schemes[j].coding_id.startswith('TruncatedFourier'):
            str_name = 'Truncated Fourier (Wide)' + f'K={imaging_schemes[j].n_codes}'
        elif imaging_schemes[j].coding_id.startswith('Gated'):
            str_name = 'Coarse Hist. (Wide)' + f'K={imaging_schemes[j].n_gates}'
            if imaging_schemes[j].pulse_width == 1:
                pass
        elif imaging_schemes[j].coding_id.startswith('Hamiltonian'):
            if imaging_schemes[j].coding_id.endswith('3') or imaging_schemes[j].coding_id.endswith('5'):
                continue
            str_name = f'SiP Hamiltonian K={imaging_schemes[j].coding_id[-1]}'
        elif imaging_schemes[j].coding_id == 'Identity':
            if imaging_schemes[j].pulse_width == 1:
                str_name = 'Full-Res. Hist. (Narrow)'
            else:
                str_name = 'Full-Res. Hist. (Wide)'
            pass
        elif imaging_schemes[j].coding_id.startswith('KTapSin'):
            if imaging_schemes[j].cw_tof is True:
                str_name = 'i-ToF Sinusoid'
            else:
                str_name = 'CoWSiP-ToF Sinusoid'

        elif imaging_schemes[j].coding_id == 'Greys':
            str_name = 'Count. Greys'
        elif imaging_schemes[j].coding_id == 'Learned':
            pass


        #k = imaging_schemes[j].coding_obj.n_functions
        k = 4

        label = get_string_name(imaging_schemes[j])
        surf = axs[0][i].plot_surface(np.log10(levels_one),np.log10(levels_two), tmp,
                               label=label, alpha=0.6,
                               edgecolors='k', lw=0.5, shade=False, antialiased=True,
                               color=get_scheme_color(imaging_schemes[j].coding_id, k, cw_tof=imaging_schemes[j].cw_tof,
                                                      constant_pulse_energy=imaging_schemes[j].constant_pulse_energy))
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d


    axs[0][i].view_init(elev=15., azim=-45)
    axs[0][i].set_xticks(np.round(np.linspace(np.min(np.log10(levels_one)), np.max(np.log10(levels_one)), num=grid_size), 1))  # Set x-axis ticks
    axs[0][i].set_yticks(np.round(np.linspace(np.min(np.log10(levels_two)), np.max(np.log10(levels_two)), num=grid_size), 1))  # Set y-axis ticks

    # Optionally, customize tick labels
    axs[0][i].set_xticklabels(np.round(np.linspace(np.min(np.log10(levels_one)), np.max(np.log10(levels_one)), num=grid_size), 1), fontsize=12)
    axs[0][i].set_yticklabels(np.round(np.linspace(np.min(np.log10(levels_two)), np.max(np.log10(levels_two)), num=grid_size), 1), fontsize=12)
    axs[0][i].set_ylabel(f'Log SBR')
    axs[0][i].set_xlabel(f'Log Photon Count')

    if 'rmse' in filename:
        axs[0][i].set_zlabel('Root Mean Sq. Error (cm)')
        # if 'photonstarved' in filename:
        #     axs[0][i].set_zlim(0, 300)
        # else:
        #     axs[0][i].set_zlim(0, 15)
    else:
        axs[0][i].set_zlabel('Mean Abs. Error (cm)')
        axs[0][i].set_zlim(0, 12)
        # if 'photonstarved' in filename:
        #     axs[0][i].set_zlim(0, 250)
        # else:
        #     axs[0][i].set_zlim(0, 12)

    #axs[0][i].legend(loc='upper right', bbox_to_anchor=(0.1, 0.8), fancybox=True)

    if 'peak030' in filename:
        axs[0][i].set_zlim(0, 30)
    elif 'peak015' in filename:
        axs[0][i].set_zlim(0, 60)
    elif 'peak005' in filename:
        axs[0][i].set_zlim(0, 250)

fig.tight_layout()

#fig.savefig(os.path.join(save_folder, 'sigma30_results_rmse_photonstarved.svg'), bbox_inches='tight', dpi=3000)
#fig.savefig(os.path.join(save_folder, 'sigma10_20_30_results_rmse.svg'), bbox_inches='tight', dpi=3000)
#fig.savefig(os.path.join(save_folder, 'sigma10_results_rmse_mae.svg'), bbox_inches='tight', dpi=3000)

#fig.savefig(os.path.join(save_folder, 'sigma1_5_10_peak030_results_rmse.svg'), bbox_inches='tight', dpi=3000)

fig.savefig('tmp.svg', bbox_inches='tight', dpi=3000)

plt.show(block=True)

# fig = go.Figure(data=arr)
# fig.show()
print('hello world')