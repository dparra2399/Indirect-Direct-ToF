# Python imports
# Library imports
import time
import os

from IPython.core import debugger

from felipe_utils.tof_utils_felipe import time2depth
from plot_figures.plot_utils import get_scheme_color
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from felipe_utils import tof_utils_felipe
import felipe_utils.research_utils.signalproc_ops as signalproc_ops
import numpy as np
from matplotlib import rc
import glob
import matplotlib.gridspec as gridspec

import matplotlib

from utils.file_utils import get_string_name

font = {'family': 'serif',
        'weight': 'bold',
        'size': 12}

rc('font', **font)

import matplotlib.pyplot as plt

breakpoint = debugger.set_trace


def get_unique_scene_ids(transient_dirpath, render_params_str):
    unique_scene_filepaths = glob.glob('{}/*{}_view-0*'.format(transient_dirpath, render_params_str))
    scene_ids = []
    for i in range(len(unique_scene_filepaths)):
        scene_filename = unique_scene_filepaths[i].split('/')[-1]
        scene_ids.append(scene_filename.split('_nr')[0])
    return scene_ids


def clip_rgb(img, max_dynamic_range=1e4):
    epsilon = 1e-7
    min_img_val = np.min(img)
    new_max_img_val = max_dynamic_range * (min_img_val + epsilon)
    # Clip all pixels with intensities larger than the max dynamic range
    img[img > new_max_img_val] = new_max_img_val
    return img


def gamma_compress(img, gamma_factor=1. / 2.2): return np.power(img, gamma_factor)


def simple_tonemap(rgb_img):
    rgb_img = clip_rgb(rgb_img)
    rgb_img = gamma_compress(rgb_img)
    return rgb_img




# Press the green button in the gutter to run the script.

n_rows = 240 // 2
n_cols = 320 // 2
n_tbins = 2000
n_samples = 2048

data_dirpath = r'Z:\Research_Users\David\sample_transient_images-20240724T164104Z-001\sample_transient_images'
transient_data_dirpath = r'{}\transient_images_{}x{}_nt-{}'.format(data_dirpath, n_rows, n_cols, n_tbins, )
rgb_data_dirpath = r'{}\rgb_images_{}x{}_nt-{}'.format(data_dirpath, n_rows, n_cols, n_tbins, )
gt_depths_data_dirpath = r'{}\depth_images_{}x{}_nt-{}'.format(data_dirpath,n_rows, n_cols, n_tbins, )



render_params_str = 'nr-{}_nc-{}_nt-{}_samples-{}'.format(n_rows, n_cols, n_tbins, n_samples)

if __name__ == '__main__':


    scene_id = 'living-room-2'
    scene_filename = r'{}_{}_view-0'.format(scene_id, render_params_str)
    print("Loading: {}".format(scene_filename))
    gt_depths_img = np.load(r'{}\{}.npy'.format(gt_depths_data_dirpath, scene_filename))
    rgb_img = np.load(r'{}\{}.npy'.format(rgb_data_dirpath, scene_filename))
    hist_img = np.load(r'{}\{}.npz'.format(transient_data_dirpath, scene_filename))['arr_0']
    # rgb_image = np.load(os.path.join(rgb_folder, filename))
    # filename = r'C:\Users\Patron\PycharmProjects\Indirect-Direct-ToF\data\horse_depth_map.npy'
    # depth_image = np.load(filename)
    #hist_img = np.pad(hist_img[..., ::2] + hist_img[..., 1::2], ((0, 0), (0, 0), (0, 24)))
    (nr, nc, n_tbins) = hist_img.shape
    depths = gt_depths_img.flatten()

    params = {}
    params['n_tbins'] = n_tbins
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 15 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['rep_tau'] = 1. / params['rep_freq']
    params['T'] = 0.1  # intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM
    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()


    # Do either average photon count
    photon_counts = 1000
    sbr = 0.1
    #peak_factors = None
    peak_factor = None
    n_codes = [8, 10, 12]
    sigma = 30

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

    schemes_num = 3

    rmse = np.zeros((len(n_codes), schemes_num+1))
    mae = np.zeros((len(n_codes), schemes_num+1))

    depth_images = np.zeros((nr, nc, len(n_codes), schemes_num+1))
    error_maps = np.zeros((nr, nc, len(n_codes), schemes_num+1))
    #hist_img = np.roll(hist_img, 12, axis=-1)
    #depths = np.argmax(hist_img, axis=-1).flatten() * tbin_depth_res

    if peak_factor is not None:
        peak_name =  f"{peak_factor:.3f}".split(".")[-1]

    for j in range(len(n_codes)):
        n_code = n_codes[j]

        irf = gaussian_pulse(np.arange(params['n_tbins']), 0, sigma, circ_shifted=True)
        params['imaging_schemes'] = [
            ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=n_code, pulse_width=1,
                                account_irf=True, h_irf=irf),
            ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                 model=os.path.join('bandlimited_models', f'n{n_tbins}_k{n_code}_sigma{sigma}'),
                                account_irf=True, h_irf=irf),
            ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=n_code, pulse_width=1, account_irf=True, h_irf=irf),

            ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

        ]

        init_coding_list(n_tbins, params, t_domain=t_domain)
        imaging_schemes = params['imaging_schemes']
        for k in range(len(imaging_schemes)):
            imaging_scheme = imaging_schemes[k]
            coding_obj = imaging_scheme.coding_obj
            coding_scheme = imaging_scheme.coding_id
            light_obj = imaging_scheme.light_obj
            light_source = imaging_scheme.light_id
            rec_algo = imaging_scheme.rec_algo

            incident = np.zeros((nr * nc, n_tbins))
            tmp = np.reshape(hist_img, (nr * nc, n_tbins))

            filt = light_obj.filtered_light_source.squeeze()
            filt /= filt.sum()

            tmp = signalproc_ops.circular_corr(filt[np.newaxis, ...], tmp, axis=-1)
            tmp[tmp < 0] = 0

            total_amb_photons = photon_counts / sbr
            tmp_irf = np.copy(irf)
            for pp in range(nr * nc):
                first = tmp[pp, :] * (photon_counts / np.sum(tmp[pp, :]).astype(np.float64))
                tmp_irf = tmp_irf * (photon_counts / np.sum(tmp_irf).astype(np.float64))
                incident[pp, :] = first + (total_amb_photons / n_tbins)
            incident = np.nan_to_num(incident, nan=0.0, posinf=0.0, neginf=0.0)

            if peak_factor is not None:
                    # peak_val = np.max(incident)
                incident = np.clip(incident, 0, (peak_factor * photon_counts) + (total_amb_photons / n_tbins))
                tmp_irf = np.clip(tmp_irf, 0, (peak_factor * photon_counts))
                if irf is not None:
                    incident = np.transpose(signalproc_ops.circular_conv(irf[:, np.newaxis], np.transpose(incident), axis=0))
                    tmp_irf = signalproc_ops.circular_conv(irf[:, np.newaxis], tmp_irf[:, np.newaxis], axis=0)
                    coding_obj.update_tmp_irf(tmp_irf)
                    coding_obj.update_C_derived_params()

                #tmp2 = np.reshape(incident, (nr, nc, n_tbins))

            coded_vals = coding_obj.encode(incident, 1).squeeze()
            #coded_vals = coding_obj.encode_no_noise(incident).squeeze()

            if coding_scheme in ['sIdentity']:
                assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
                decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                                   rec_algo_id=rec_algo) * tbin_depth_res
            else:
                decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res


            error_maps[:, :, j, k] = np.abs(np.reshape(decoded_depths, (nr, nc)) - np.reshape(depths, (nr, nc)))
            depth_images[:, :, j, k] = np.reshape(decoded_depths, (nr, nc))
            errors = np.abs(decoded_depths - depths.flatten()) * depth_res
            error_metrix = calc_error_metrics(errors)
            rmse[j, k] = error_metrix['rmse']
            mae[j, k] = error_metrix['mae']

    n_codes_size = len(n_codes)
    fig = plt.figure(figsize=(25, 10))
    gs = gridspec.GridSpec(n_codes_size, schemes_num+1, figure=fig, hspace=0.05, wspace=0.05)

    for j in range(n_codes_size):
        for k in range(schemes_num):
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[j, k],
                                                        width_ratios=[1, 1], hspace=0, wspace=0.05)

            ax_low = fig.add_subplot(inner_gs[0, 0])
            ax_high = fig.add_subplot(inner_gs[0, 1])

            depth_im = ax_low.imshow(depth_images[:,:, j, k], vmin=depths.min(), vmax=depths.max())
            error_im = ax_high.imshow(error_maps[:, :, j, k] * 100, vmin=0, vmax=30)

            ax_low.text(
                2, 2,  # x, y pixel coordinates
                f"MAE: {mae[j, k] / 10:.2f} cm \nRMSE {rmse[j, k] / 10: .2f} cm",  # your text
                color='white',
                fontsize=10,
                ha='left', va='top',  # align text relative to point
                bbox=dict(facecolor='black', alpha=0.5, pad=2)  # optional background box
            )

            if j == n_codes_size - 1:
                cbar_im = fig.colorbar(depth_im, ax=ax_low, orientation='horizontal', label='Depth (meters)', pad=0.04)
                cbar_error = fig.colorbar(error_im, ax=ax_high, orientation='horizontal', label='Error (cm)', pad=0.04)

            ax_low.get_xaxis().set_ticks([])
            ax_low.get_yaxis().set_ticks([])
            ax_high.get_xaxis().set_ticks([])
            ax_high.get_yaxis().set_ticks([])

            print(f'Scheme: {imaging_schemes[k].coding_id}, RMSE: {rmse[j, k] / 10: .2f} cm, MAE: {mae[j, k] / 10:.2f} cm')

            if j == 0:

                if imaging_schemes[k].coding_id.startswith('TruncatedFourier'):
                    str_name = 'Truncated Fourier'
                elif imaging_schemes[k].coding_id == 'Identity':
                    str_name = 'FRH'
                elif imaging_schemes[k].coding_id == 'Greys':
                    str_name = 'Count. Gray'
                elif imaging_schemes[k].coding_id.startswith('Learned'):
                    str_name = 'Optimized'

                #ax_low.set_title(str_name)

        if j == 0:
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[j, k + 1], width_ratios=[1, 1], hspace=0,
                                                        wspace=0.05)

            ax_rgb = fig.add_subplot(inner_gs[0, 0])
            ax_rgb.imshow(gt_depths_img)
            ax_rgb.get_xaxis().set_ticks([])
            ax_rgb.get_yaxis().set_ticks([])

            # Optional: hide the second (dummy) axis
            ax_dummy = fig.add_subplot(inner_gs[0, 1])
            ax_dummy.imshow(rgb_img)
            ax_dummy.get_xaxis().set_ticks([])
            ax_dummy.get_yaxis().set_ticks([])

        elif j == 1:
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[j, k + 1], width_ratios=[1, 1], hspace=0,
                                                        wspace=0.05)

            ax_low = fig.add_subplot(inner_gs[0, 0])
            ax_high = fig.add_subplot(inner_gs[0, 1])

            depth_im = ax_low.imshow(depth_images[:,:, j, k+1], vmin=depths.min(), vmax=depths.max())
            ax_low.plot(100, 100, 'o', color='blue', markersize=6)
            ax_low.plot(110, 90, 'o', color='orange', markersize=6)


            error_im = ax_high.imshow(error_maps[:, :, j, k+1] * 100, vmin=0, vmax=30)

            ax_low.get_xaxis().set_ticks([])
            ax_low.get_yaxis().set_ticks([])
            ax_high.get_xaxis().set_ticks([])
            ax_high.get_yaxis().set_ticks([])

            cbar_im = fig.colorbar(depth_im, ax=ax_low, orientation='horizontal', label='Depth (meters)', pad=0.04)
            cbar_error = fig.colorbar(error_im, ax=ax_high, orientation='horizontal', label='Error (cm) and RGB', pad=0.04)

            ax_low.text(
                2, 2,  # x, y pixel coordinates
                f"MAE: {mae[j, k+1] / 10:.2f} cm \nRMSE {rmse[j, k+1] / 10: .2f} cm",  # your text
                color='white',
                fontsize=10,
                ha='left', va='top',  # align text relative to point
                bbox=dict(facecolor='black', alpha=0.5, pad=2)  # optional background box
            )
        elif j == 2:
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gs[j, k + 1], width_ratios=[1, 1], hspace=0, wspace=0.05
            )

            ax_low = fig.add_subplot(inner_gs[0, 0])
            ax_high = fig.add_subplot(inner_gs[0, 1])

            # Plot the histograms
            ax_low.plot(hist_img[100, 100, :], color='blue')
            ax_high.plot(hist_img[90, 110, :], color='orange')

            ax_low.set_xticks((np.linspace(0, 2000, 3)))
            ax_low.set_xticklabels((np.linspace(0, 2000, 3) * tbin_res * 1e9).astype(int))
            ax_low.set_yticks([])
            ax_low.set_xlabel('Time (ns)')
            ax_low.set_ylabel('Counts')
            ax_low.set_box_aspect(1 / 1.5)  # match 1 row : 2 cols like your images


            ax_high.set_xticks((np.linspace(0, 2000, 3)))
            ax_high.set_xticklabels((np.linspace(0, 2000, 3) * tbin_res * 1e9).astype(int))
            ax_high.set_yticks([])
            ax_high.set_xlabel('Time (ns)')
            ax_high.set_box_aspect(1 / 1.5)  # match 1 row : 2 cols like your images


    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    #fig.savefig(f'bit_depth_grid_peak.svg', bbox_inches='tight')
    plt.show(block=True)

print('YAYYY')
