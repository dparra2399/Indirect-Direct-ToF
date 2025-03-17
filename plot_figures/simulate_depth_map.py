# Python imports
# Library imports
import time
import os

from IPython.core import debugger

from plot_figures.plot_utils import get_scheme_color
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
import numpy as np
from matplotlib import rc
import matplotlib


font = {'family': 'serif',
        'weight': 'bold',
        'size': 12}

rc('font', **font)

import matplotlib.pyplot as plt

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

    params['imaging_schemes'] = [
        # ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=32),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=4, pulse_width=1, account_irf=True),
        ImagingSystemParams('Learned', 'Learned', 'zncc', model=os.path.join('bandlimited_peak_models', 'n1024_k4_mae')),
        ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=4, pulse_width=1, account_irf=True),

        # ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, freq_window=0.05),

    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    #filename = 'veach-bidir_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    #filename = 'cbox_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    #filename = 'breakfast-hall_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    #filename = 'hot-living_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    filename = 'staircase_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    depth_image = np.load(os.path.join(depth_folder, filename))
    rgb_image = np.load(os.path.join(rgb_folder, filename))
    (nr, nc) = depth_image.shape
    depths = depth_image.flatten()
    print(f'Max Depth {depths.max()}')

    # Do either average photon count
    peak_photon_counts = [2000]
    ambient_counts = [0.1]
    peak_factor = 0.005

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

    depth_images = np.zeros((len(peak_photon_counts), nr, nc, len(params['imaging_schemes'])))
    byte_sizes = np.zeros((len(peak_photon_counts), len(params['imaging_schemes'])))
    rmse = np.zeros((len(peak_photon_counts), len(params['imaging_schemes'])))
    mae = np.zeros((len(peak_photon_counts), len(params['imaging_schemes'])))

    tic = time.perf_counter()

    for j in range(len(peak_photon_counts)):
        photon_count = peak_photon_counts[j]
        ambient = ambient_counts[j]
        for i in range(len(imaging_schemes)):
            imaging_scheme = imaging_schemes[i]
            coding_obj = imaging_scheme.coding_obj
            coding_scheme = imaging_scheme.coding_id
            light_obj = imaging_scheme.light_obj
            light_source = imaging_scheme.light_id
            rec_algo = imaging_scheme.rec_algo

            incident = light_obj.simulate_average_photons(photon_count, ambient, peak_factor=peak_factor)

            coded_vals = coding_obj.encode(incident, trials).squeeze()

            #coded_vals = coding_obj.encode_no_noise(incident).squeeze()

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

    fig, axs = plt.subplots(len(peak_photon_counts)*2, len(params['imaging_schemes'])+1, squeeze=False)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    counter1 = 0
    counter2 = 1

    axs[0][0].set_ylabel('Depth Map')
    axs[1][0].set_ylabel('Depth Errors (mm)')

    for j in range(0, len(peak_photon_counts)):
        for i in range(len(params['imaging_schemes'])):

            scheme = params['imaging_schemes'][i]
            depth_map = depth_images[j, :, :, i]

            #depth_map[depth_map < 1/2*np.min(depth_image)] = np.nan
            #depth_map[depth_map > 2*np.max(depth_image)] = np.nan

            error_map = np.abs(depth_map - depth_image)

            depth_im = axs[counter1][i].imshow(depth_map,
                                               vmin=0.8*np.min(depth_image), vmax=1.2*np.max(depth_image))

            for spine in axs[counter1][i].spines.values():
                spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
                spine.set_linewidth(4)

            error_im = axs[counter2][i].imshow(error_map, vmin=0, vmax=0.5)

            for spine in axs[counter2][i].spines.values():
                spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
                spine.set_linewidth(4)

            axs[counter1][i].get_xaxis().set_ticks([])
            axs[counter1][i].get_yaxis().set_ticks([])
            axs[counter2][i].get_xaxis().set_ticks([])
            axs[counter2][i].get_yaxis().set_ticks([])
            #if counter == 2:
            axs[counter2][i].set_xlabel(f'RMSE: {rmse[j, i] / 10: .2f} cm \n MAE: {mae[j, i] / 10:.2f} cm')
            axs[0][i].set_title(scheme.coding_id)
            print(f'Scheme: {scheme.coding_id}, RMSE: {rmse[j, i] / 10: .2f} cm, MAE: {mae[j, i] / 10:.2f} cm')
        counter1 += 2
        counter2 += 2

    axs[0, -1].axis('off')
    axs[1, -1].axis('off')
    cbar_im = fig.colorbar(depth_im, ax=axs[0, -1], orientation='vertical', label='Depth (meters)')
    cbar_error = fig.colorbar(error_im, ax=axs[1, -1], orientation='vertical', label='Error (meters)')

    axs[0, -1].legend()
    axs[1, -1].legend()
    #fig.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    #fig.savefig('Z:\\Research_Users\\David\\paper figures\\figure6a.svg', bbox_inches='tight', dpi=3000)
    plt.show()
    print()
print('YAYYY')
