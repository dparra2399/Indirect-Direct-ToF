# Python imports
# Library imports
import time
import os

from IPython.core import debugger
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from spad_toflib import spad_tof_utils
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
import numpy as np
from spad_toflib.emitted_lights import GaussianTIRF
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils.file_utils import get_string_name
import matplotlib
import matplotlib.pyplot as plt

breakpoint = debugger.set_trace

depth_folder = '/Users/Patron/Downloads/sample_transient_images/depth_images_240x320_nt-2000/'
rgb_folder = '/Users/Patron/Downloads/sample_transient_images/rgb_images_240x320_nt-2000/'
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    params = {}
    params['n_tbins'] = 1024
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 10 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['rep_tau'] = 1. / params['rep_freq']
    params['T'] = 0.1  # intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM

    square_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_square_10mhz.npy')
    pulse_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_pulse_10mhz.npy')

    pulse_width = 8e-9
    tbin_res = params['rep_tau'] / params['n_tbins']
    sigma = int(pulse_width / tbin_res)

    params['imaging_schemes'] = [
        ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                            duty=1. / 15., freq_window=0.10, binomial=True, gated=True,
                            total_laser_cycles=100_000),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1,
                            binomial=True, gated=True, total_laser_cycles=100_000),
        ImagingSystemParams('Gated', 'Gaussian', 'linear', n_gates=16, pulse_width=10,
                            binomial=True, gated=True, total_laser_cycles=100_000)
    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    filename = 'living-room-3_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    depth_image = np.load(os.path.join(depth_folder, filename))
    rgb_image = np.load(os.path.join(rgb_folder, filename))
    (nr, nc) = depth_image.shape
    depths = depth_image.flatten()

    # Do either average photon count
    photon_count = (10 ** 4)
    sbr = 0.1
    laser_cycles = [3e5, 9e5]

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

    toc = time.perf_counter()

    fig, axs = plt.subplots(len(laser_cycles), len(params['imaging_schemes']), figsize=(8, 8), squeeze=False)

    counter = 0
    for j in range(0, len(laser_cycles)):
        #axs[counter][0].set_ylabel('Depth Map')
        axs[counter][0].set_ylabel(r'$\bf{Log Laser Cycles:}$' + f' {int(laser_cycles[j])}',
                                   labelpad=10)
        for i in range(len(params['imaging_schemes'])):
            scheme = params['imaging_schemes'][i]
            depth_map = depth_images[j, :, :, i]

            image_mask = np.zeros_like(depth_map)
            image_mask[depth_map < np.min(depth_image)] = 1
            image_mask[depth_map > np.max(depth_image)] = 1


            depth_map[depth_map < np.min(depth_image)] = np.mean(depth_image)
            depth_map[depth_map > np.max(depth_image)] = np.mean(depth_image)

            error_map = np.abs(depth_map - depth_image)

            num_outliers = (error_map[error_map > 0.3]).size
            percent_outliers = num_outliers / depth_image.size


            axs[counter][i].imshow(depth_map, cmap='hsv')
            axs[counter][i].imshow(image_mask, cmap='binary', alpha=0.9 * (image_mask > 0))

            # axs[counter+1][i].imshow(error_map, vmin=0, vmax=1.0, cmap='hot')

            if counter == 0:
                axs[counter][i].set_title(f'{get_string_name(scheme)}')
            axs[counter][i].get_xaxis().set_ticks([])
            axs[counter][i].get_yaxis().set_ticks([])
            # axs[counter+1][i].get_xaxis().set_ticks([])
            # axs[counter+1][i].get_yaxis().set_ticks([])
            #if counter == 2:
            axs[counter][i].set_xlabel(f'Outliers: {percent_outliers * 100: .1f}% \n '
                                 f'RMSE: {rmse[j, i] / 10: .2f} cm \n '
                                 f'MAE: {mae[j, i] / 10: .2f} cm ')
        counter += 1
    fig.tight_layout()
    fig.savefig('figure5.jpg', bbox_inches='tight')
    plt.show()
#plt.show()
print()
print('YAYYY')
