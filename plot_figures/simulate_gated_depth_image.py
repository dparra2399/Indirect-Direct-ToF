# Python imports
# Library imports
import time
import os

from IPython.core import debugger

from plot_figures.plot_utils import get_scheme_color
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics
import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor

from matplotlib import rc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')

font = {'family': 'serif',
        'weight': 'bold',
        'size': 7}

rc('font', **font)

breakpoint = debugger.set_trace

depth_folder = r'Z:\\Research_Users\\David\\sample_transient_images-20240724T164104Z-001\\sample_transient_images\\depth_images_240x320_nt-2000'
rgb_folder =  r'Z:\\Research_Users\\David\\sample_transient_images-20240724T164104Z-001\\sample_transient_images\\rgb_images_240x320_nt-2000'
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    params = {}
    params['n_tbins'] = 1024 * 2
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 10 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['rep_tau'] = 1. / params['rep_freq']
    params['T'] = 0.1  # intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM



    params['imaging_schemes'] = [
        ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
                            binomial=True, gated=True,
                            total_laser_cycles=100_000),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1,
                            binomial=True, gated=True, total_laser_cycles=100_000),
        ImagingSystemParams('Gated', 'Gaussian', 'linear', n_gates=32, pulse_width=20,
                             binomial=True, gated=True, total_laser_cycles=100_000),
    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]
    print(f'max depth: {params["dMax"]} meters')
    print()

    #filename = 'living-room-3_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    #filename = 'cbox_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    filename = 'staircase_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    depth_image = np.load(os.path.join(depth_folder, filename))
    rgb_image = np.load(os.path.join(rgb_folder, filename))
    (nr, nc) = depth_image.shape
    depths = depth_image.flatten()
    # Do either average photon count
    photon_count = (10 ** 3)
    sbr = 0.1
    peak_photon_count = None
    ambient_count = 10
    laser_cycles = [7e6]

    n_tbins = params['n_tbins']
    mean_beta = params['meanBeta']
    tau = params['rep_tau']
    depth_res = params['depth_res']
    t = params['T']
    trials = params['trials']
    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))

    print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')
    print()

    init_coding_list(n_tbins, depths, params, t_domain=t_domain)
    imaging_schemes = params['imaging_schemes']

    depth_images = np.zeros((len(laser_cycles), nr, nc, len(params['imaging_schemes'])))
    byte_sizes = np.zeros((len(laser_cycles), len(params['imaging_schemes'])))
    rmse = np.zeros((len(laser_cycles), len(params['imaging_schemes'])))
    mae = np.zeros((len(laser_cycles), len(params['imaging_schemes'])))

    tic = time.perf_counter()

    for j in range(len(laser_cycles)):
        laser_cycle = laser_cycles[j]
        for i in range(len(imaging_schemes)):
            imaging_scheme = imaging_schemes[i]
            coding_obj = imaging_scheme.coding_obj
            coding_scheme = imaging_scheme.coding_id
            light_obj = imaging_scheme.light_obj
            light_source = imaging_scheme.light_id
            rec_algo = imaging_scheme.rec_algo

            if peak_photon_count is not None:
                incident = light_obj.simulate_peak_photons(peak_photon_count, ambient_count)
            else:
                incident = light_obj.simulate_average_photons(photon_count, sbr)

            coding_obj.set_laser_cycles(laser_cycle)

            coded_vals = coding_obj.encode(incident, trials).squeeze()

            if coding_scheme in ['Identity']:
                assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
                decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                                   rec_algo_id=rec_algo) * tbin_depth_res
            else:
                decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

            depth_images[j, :, :, i] = np.reshape(decoded_depths, (nr, nc))
            byte_sizes[j, i] = np.squeeze(coded_vals).size * np.squeeze(coded_vals).itemsize
            errors = np.abs(decoded_depths - depths[np.newaxis, :]) * depth_res
            error_metrix = calc_error_metrics(errors)

            rmse[j, i] = error_metrix['rmse']
            mae[j, i] = error_metrix['mae']
            print(f'rmse: {error_metrix['rmse']}, mae: {error_metrix['mae']}')

    toc = time.perf_counter()


    fig, axs = plt.subplots(len(laser_cycles)*2, len(params['imaging_schemes']), figsize=(4.4, 2.6), squeeze=False)
    plt.subplots_adjust(hspace=0.05, wspace=0.01)

    counter1 = 0
    counter2 = 1

    for j in range(0, len(laser_cycles)):
        #axs[counter][0].set_ylabel('Depth Map')
        #axs[counter1][0].set_ylabel(r'$\bf{\Delta t:}$' + f': {peak_photon_counts[j]} \n'
                                   #r'$\bf{\Phi^{bkg}:}$' + f'{ambient_counts[j]}:', rotation='horizontal',
                                   #labelpad=20)
        for i in range(len(params['imaging_schemes'])):
            scheme = params['imaging_schemes'][i]
            depth_map = depth_images[j, :, :, i]


            #depth_map[depth_map < np.min(depth_image)] = np.mean(depth_image)
            #depth_map[depth_map > np.max(depth_image)] = np.mean(depth_image)

            error_map = np.abs(depth_map - depth_image)

            num_outliers = (error_map[error_map > 0.3]).size
            percent_outliers = num_outliers / depth_image.size


            depth_im = axs[counter1][i].imshow(depth_map,
                                               vmin=0.8*np.min(depth_image), vmax=1.2*np.max(depth_image))

            for spine in axs[counter1][i].spines.values():
                spine.set_edgecolor(
                    get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
                spine.set_linewidth(4)

            error_im = axs[counter2][i].imshow(error_map, vmin=0, vmax=0.5)

            for spine in axs[counter2][i].spines.values():
                spine.set_edgecolor(
                    get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
                spine.set_linewidth(4)

            # if counter1 == 0:
            #     str_name = ''
            #     if scheme.coding_id.startswith('TruncatedFourier'):
            #         str_name = 'Truncated Fourier \n (Wide)'
            #         axs[0][i].set_title(str_name)
            #     elif scheme.coding_id.startswith('Gated'):
            #         str_name = 'Coarse Hist. \n (Wide)'
            #         axs[0][i].set_title(str_name)
            #     elif scheme.coding_id.startswith('Hamiltonian'):
            #         str_name = 'SiP Hamiltonian' + '\n' + r'(\textcolor{red}{Proposed})'
            #         axs[0][i].set_title(str_name, color='red')
            #     elif scheme.coding_id == 'Identity':
            #         if scheme.pulse_width == 1:
            #             str_name = 'Full-Res. Hist. \n (Narrow)'
            #             axs[0][i].set_title(str_name)
            #         else:
            #             str_name = 'Full-Res. Hist. \n (Wide)'
            #             axs[0][i].set_title(str_name)

            axs[counter1][i].get_xaxis().set_ticks([])
            axs[counter1][i].get_yaxis().set_ticks([])
            axs[counter2][i].get_xaxis().set_ticks([])
            axs[counter2][i].get_yaxis().set_ticks([])
            #if counter == 2:
            #axs[counter2][i].set_xlabel(f'RMSE: {rmse[j, i] / 10: .2f} cm')
            print(f'Scheme: {scheme.coding_id}, RMSE: {rmse[j, i] / 10: .2f} cm, MAE: {mae[j, i] / 10:.2f} cm')

        counter1 += 2
        counter2 += 2

    cbar_im = fig.colorbar(depth_im, ax=axs[0, :], orientation='vertical')
    cbar_error = fig.colorbar(error_im, ax=axs[1, :], orientation='vertical')

    fig.tight_layout()
    fig.savefig('Z:\\Research_Users\\David\\paper figures\\figure7b.svg', bbox_inches='tight', dpi=3000)
    plt.show()
#plt.show()
print()
print('YAYYY')
