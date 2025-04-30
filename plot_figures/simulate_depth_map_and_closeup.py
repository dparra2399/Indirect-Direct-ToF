# Python imports
# Library imports
import time
import os

from IPython.core import debugger

from plot_figures.plot_utils import get_scheme_color
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from felipe_utils import tof_utils_felipe
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

    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 5, circ_shifted=True)
    params['imaging_schemes'] = [
        # ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=32),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1, account_irf=True, h_irf=irf),
        #ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma5_peak015_counts1000'),
                           account_irf=True, h_irf=irf),

        #ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),


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
    # filename = 'staircase_nr-240_nc-320_nt-2000_samples-2048_view-0.npy'
    # depth_image = np.load(os.path.join(depth_folder, filename))
    # rgb_image = np.load(os.path.join(rgb_folder, filename))
    filename = r'C:\Users\Patron\PycharmProjects\Indirect-Direct-ToF\data\cow_depth_map.npy'
    depth_image = np.load(filename)
    (nr, nc) = depth_image.shape
    depths = depth_image.flatten()
    print(f'Max Depth {depths.max()}')

    # Do either average photon count
    peak_photon_counts = [1000]
    ambient_counts = [0.1]
    peak_factor = 0.015

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
    error_maps = np.zeros((len(peak_photon_counts), nr, nc, len(params['imaging_schemes'])))

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


            if 'cow' == filename.split('.')[-2].split("_")[0].split(os.path.sep)[-1]:
                mask = plt.imread(r'C:\Users\Patron\PycharmProjects\Indirect-Direct-ToF\data\cow.png')
                mask = mask[..., -1]
                depths_masked = mask.flatten() * depths
                decoded_depths = mask.flatten() * decoded_depths
            elif 'horse' == filename.split('.')[-2].split("_")[0].split(os.path.sep)[-1]:
                mask = plt.imread(r'C:\Users\Patron\PycharmProjects\Indirect-Direct-ToF\data\horse.png')
                mask = mask[..., -1]
                depths_masked = mask.flatten() * depths
                decoded_depths = mask.flatten() * decoded_depths
            else:
                depths_masked = depths

            normalized_decoded_depths = np.copy(decoded_depths)
            normalized_decoded_depths[normalized_decoded_depths == 0] = np.nan
            #vmin = np.nanmean(normalized_decoded_depths) - 30
            #vmax = np.nanmean(normalized_decoded_depths) + 30
            vmin = 12#depths.min() - 20
            vmax = 25 #depths.max() + 20
            normalized_decoded_depths[normalized_decoded_depths > vmax] = np.nan
            normalized_decoded_depths[normalized_decoded_depths < vmin] = np.nan


            error_maps[j, :, :, i] = np.abs(np.reshape(normalized_decoded_depths, (nr, nc)) - np.reshape(depths, (nr, nc)))
            depth_images[j, :, :, i] = np.reshape(normalized_decoded_depths, (nr, nc))
            errors = np.abs(normalized_decoded_depths - depths_masked.flatten()[np.newaxis, :]) * depth_res
            error_metrix = calc_error_metrics(errors[~np.isnan(errors)])


            rmse[j, i] = error_metrix['rmse']
            mae[j, i] = error_metrix['mae']

    toc = time.perf_counter()

    fig, axs = plt.subplots(len(peak_photon_counts)*3, len(params['imaging_schemes'])+1, squeeze=False, figsize=(12, 5))

    counter1 = 0
    counter2 = 1
    counter3 = 2

    axs[0][0].set_ylabel('Depth Map')

    for j in range(0, len(peak_photon_counts)):
        for i in range(len(params['imaging_schemes'])):

            scheme = params['imaging_schemes'][i]
            depth_map = depth_images[j, :, :, i]
            tmp = np.copy(depth_map)
            tmp[np.isnan(tmp)] = 0


            #depth_map[depth_map < 1/2*np.min(depth_image)] = np.nan
            #depth_map[depth_map > 2*np.max(depth_image)] = np.nan

            error_map = error_maps[j, :, :, i] * 10
            error_map[np.isnan(error_map)] = 0


            depth_map_small = tmp[1:90, 1:70]
            error_map_small = error_map[1:90, 1:70]
            depth_im = axs[counter1][i].imshow(tmp,
                                               vmin=np.nanmin(depth_images), vmax=np.nanmax(depth_images))


            rect = matplotlib.patches.Rectangle((1, 1), 70, 90, linewidth=2, edgecolor='red', facecolor='none')
            axs[counter1][i].add_patch(rect)

            for spine in axs[counter1][i].spines.values():
                #spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
                spine.set_linewidth(2)

            depth_im_small = axs[counter2][i].imshow(depth_map_small,
                                                vmin=np.nanmin(depth_images), vmax=np.nanmax(depth_images))
            error_im_small = axs[counter3][i].imshow(error_map_small,
                                                     vmin=0, vmax=11)

            for spine in axs[counter2][i].spines.values():
                #spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
                spine.set_linewidth(2)

            for spine in axs[counter3][i].spines.values():
                # spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
                spine.set_linewidth(2)

            axs[counter1][i].get_xaxis().set_ticks([])
            axs[counter1][i].get_yaxis().set_ticks([])
            axs[counter2][i].get_xaxis().set_ticks([])
            axs[counter2][i].get_yaxis().set_ticks([])
            axs[counter3][i].get_xaxis().set_ticks([])
            axs[counter3][i].get_yaxis().set_ticks([])
            #if counter == 2:
            axs[counter2][i].set_xlabel(f'RMSE: {rmse[j, i] / 10: .2f} cm \n MAE: {mae[j, i] / 10:.2f} cm')
            str_name = ''
            if imaging_schemes[i].coding_id.startswith('TruncatedFourier'):
                str_name = 'Trunc. Fourier'
            elif imaging_schemes[i].coding_id.startswith('Gated'):
                str_name = 'Coarse Hist. (Wide)' + f'K={imaging_schemes[i].n_gates}'
            elif imaging_schemes[i].coding_id.startswith('Hamiltonian'):
                str_name = f'SiP Hamiltonian K={imaging_schemes[i].coding_id[-1]}'
            elif imaging_schemes[i].coding_id == 'Identity':
                str_name = 'Full-Res. Hist.'
            elif imaging_schemes[i].coding_id.startswith('KTapSin'):
                if imaging_schemes[i].cw_tof is True:
                    str_name = 'i-ToF Sinusoid'
                else:
                    str_name = 'CoWSiP-ToF Sinusoid'

            elif imaging_schemes[i].coding_id == 'Greys':
                str_name = 'Greys'
            elif imaging_schemes[i].coding_id.startswith('Learned'):
                str_name = 'Learned'

            axs[0][i].set_title(str_name)
            print(f'Scheme: {scheme.coding_id}, RMSE: {rmse[j, i] / 10: .2f} cm, MAE: {mae[j, i] / 10:.2f} cm')
        counter1 += 3
        counter2 += 3
        counter3 += 3

    axs[0, -1].axis('off')
    axs[1, -1].axis('off')
    cbar_im = fig.colorbar(depth_im, ax=axs.flatten(), orientation='vertical', label='Depth (meters)')
    axs[0, -1].legend()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    fig.savefig('Z:\\Research_Users\\David\\Learned Coding Functions Paper\\cow_tease.svg', bbox_inches='tight', dpi=3000)
    plt.show()
    print()
print('YAYYY')
