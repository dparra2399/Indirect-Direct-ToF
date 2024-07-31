# Python imports
# Library imports
import time
import os

from IPython.core import debugger
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from spad_toflib import spad_tof_utils
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
import numpy as np
from spad_toflib.emitted_lights import GaussianTIRF
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils.file_utils import get_string_name
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

breakpoint = debugger.set_trace

depth_folder = '/Users/Patron/Downloads/sample_transient_images/depth_images_240x320_nt-2000/'
rgb_folder = '/Users/Patron/Downloads/sample_transient_images/rgb_images_240x320_nt-2000/'
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    params = {}
    params['n_tbins'] = 2000
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
        ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
                            duty=1. / 4., freq_window=0.10),
        ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                            duty=1. / 4., freq_window=0.10),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=5),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=sigma),
        ImagingSystemParams('Gated', 'Gaussian', 'linear', n_gates=100, pulse_width=sigma)
    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    filename = 'office_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    depth_image = np.load(os.path.join(depth_folder, filename))
    rgb_image = np.load(os.path.join(rgb_folder, filename))
    (nr, nc) = depth_image.shape
    depths = depth_image.flatten()

    # Do either average photon count
    photon_count = (10 ** 6)
    sbr = 1
    # Or peak photon count
    peak_photon_count = 10
    ambient_count = 5

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

    if peak_photon_count is not None:
        print(f'Peak Photon count : {peak_photon_count}')
        print()

    init_coding_list(n_tbins, depths, params, t_domain=t_domain)
    imaging_schemes = params['imaging_schemes']

    depth_images = np.zeros((nr, nc, len(params['imaging_schemes'])))
    byte_sizes = np.zeros(len(params['imaging_schemes']))
    tic = time.perf_counter()

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

        coded_vals = coding_obj.encode(incident, trials).squeeze()

        if coding_scheme in ['Identity']:
            assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
            decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                               rec_algo_id=rec_algo) * tbin_depth_res
        else:
            decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

        depth_images[:, :, i] = np.reshape(decoded_depths, (nr, nc))
        byte_sizes[i] = np.squeeze(coded_vals).size * np.squeeze(coded_vals).itemsize
        imaging_scheme.mean_absolute_error = spad_tof_utils.compute_metrics(depths, decoded_depths) * depth_res

    toc = time.perf_counter()

    fig, axs = plt.subplots(1, len(params['imaging_schemes']) + 1, figsize=(300, 300), squeeze=False)

    axs[0][0].imshow(depth_image, cmap='hsv')
    axs[0][0].set_title('Ground Truth')
    axs[0][0].get_xaxis().set_ticks([])
    axs[0][0].get_yaxis().set_ticks([])
    # axs[1][1].imshow(rgb_image)
    # axs[1][1].get_xaxis().set_ticks([])
    # axs[1][1].get_yaxis().set_ticks([])

    for i in range(len(params['imaging_schemes'])):
        scheme = params['imaging_schemes'][i]
        depth_map = depth_images[:, :, i]

        image_mask = np.zeros_like(depth_map)
        image_mask[depth_map < np.min(depth_image)] = 1
        image_mask[depth_map > np.max(depth_image)] = 1

        depth_map[depth_map < np.min(depth_image)] = np.mean(depth_image)
        depth_map[depth_map > np.max(depth_image)] = np.mean(depth_image)

        axs[0][i + 1].imshow(depth_map, cmap='hsv')
        axs[0][i + 1].imshow(image_mask, cmap='binary', alpha=0.9*(image_mask>0))
        # axs[1][i+1].imshow(rgb_image)
        # axs[1][i + 1].imshow(image_mask, cmap='hsv', alpha=0.9*(image_mask>0))

        axs[0][i + 1].set_title(f'{get_string_name(scheme)} \n Data Rate = {byte_sizes[i] * 1e-9 * 30: .3f} GB/s')
       # axs[0][i+1].set_title('penis', y=-0.1)
        axs[0][i + 1].get_xaxis().set_ticks([])
        axs[0][i + 1].get_yaxis().set_ticks([])
        axs[0][i + 1].set_xlabel(f'MAE = {scheme.mean_absolute_error / 10: .3f} cm')
        # axs[1][i+1].get_xaxis().set_ticks([])
        # axs[1][i + 1].get_yaxis().set_ticks([])

        print(f"MAE {scheme.coding_id}: {scheme.mean_absolute_error / 10: .3f} cm,", end='')
        print(f' Data Rate = {byte_sizes[i] * 1e-9 * 30: .3f} GB/s')


    fig.tight_layout()
    fig.savefig('foo.jpg', bbox_inches='tight')
    plt.show()
#plt.show()
print()
print('YAYYY')
