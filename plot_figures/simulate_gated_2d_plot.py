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
# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor

from utils.file_utils import get_string_name
from matplotlib import rc
import matplotlib
import matplotlib.pyplot as plt
import plot_utils
matplotlib.use('TkAgg')

rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')

font = {'family': 'serif',
        'size': 12}

rc('font', **font)

breakpoint = debugger.set_trace

depth_folder = r'Z:\\Research_Users\\David\\sample_transient_images-20240724T164104Z-001\\sample_transient_images\\depth_images_240x320_nt-2000'
rgb_folder =  r'Z:\\Research_Users\\David\\sample_transient_images-20240724T164104Z-001\\sample_transient_images\\rgb_images_240x320_nt-2000'
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    params = {}
    params['n_tbins'] = 1024
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 5 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['rep_tau'] = 1. / params['rep_freq']
    params['T'] = 0.1  # intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM

    #square_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_square_10mhz.npy')
    #pulse_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_pulse_10mhz.npy')

    pulse_width = 8e-9
    tbin_res = params['rep_tau'] / params['n_tbins']
    sigma = int(pulse_width / tbin_res)

    params['imaging_schemes'] = [
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1,
                            binomial=True, gated=True, total_laser_cycles=100_000),
        ImagingSystemParams('Gated', 'Gaussian', 'linear', n_gates=64, pulse_width=sigma,
                             binomial=True, gated=True, total_laser_cycles=100_000),
        ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                            duty=1. / 4., freq_window=0.10, binomial=True, gated=True,
                            total_laser_cycles=100_000),
    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    filename = 'cbox_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    depth_image = np.load(os.path.join(depth_folder, filename))
    rgb_image = np.load(os.path.join(rgb_folder, filename))
    (nr, nc) = depth_image.shape
    line = int(nr//2)
    depths = depth_image[line, :].flatten()

    # Do either average photon count
    photon_count = (10 ** 4)
    sbr = 0.1
    peak_photon_count = 10
    ambient_count = 10

    laser_cycle = 2e6

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

    tic = time.perf_counter()

    fig, axs = plt.subplots()

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

        errors = np.abs(decoded_depths - depths[np.newaxis, :]) * depth_res
        error_metrix = calc_error_metrics(errors)


        for pp in range(decoded_depths.shape[-1]):
            axs.plot([pp, pp], [depth_image[line, pp], decoded_depths[pp]],
                     color=plot_utils.get_scheme_color(coding_scheme, pw=imaging_scheme.pulse_width),
                     linewidth=3.0)

        print(f'rmse: {error_metrix['rmse']}, mae: {error_metrix['mae']}')

    axs.plot(np.linspace(0, nc, nc), depth_image[line, :], color='red')
    axs.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    fig.tight_layout()
    axs.set_ylim(2,7)
    #fig.savefig('Z:\\Research_Users\\David\\paper figures\\figure7a.svg', bbox_inches='tight')
    plt.show()
print()
print('YAYYY')
