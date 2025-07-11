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

# for k in [8, 10, 14]:
#     coding = np.load(f'/Users/Patron/PycharmProjects/Indirect-Direct-ToF/learned_codes/bandlimited_models/n2048_k{k}_sigma30/coded_model.npy')
#     coding = np.pad(coding, ((0, 48), (0, 0)))
#     np.save(f'/Users/Patron/PycharmProjects/Indirect-Direct-ToF/learned_codes/bandlimited_models/n2048_k{k}_sigma30/coded_model.npy', coding)
#     illum = np.load(f'/Users/Patron/PycharmProjects/Indirect-Direct-ToF/learned_codes/bandlimited_models/n2048_k{k}_sigma30/illum_model.npy')
#     illum = np.pad(illum, ((0, 0), (0, 48)))
#     np.save(f'/Users/Patron/PycharmProjects/Indirect-Direct-ToF/learned_codes/bandlimited_models/n2048_k{k}_sigma30/illum_model.npy', illum)
#



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
factor = 1
n_rows = 240 // factor
n_cols = 320 // factor
n_tbins = 2000
n_samples = 2048

#data_dirpath = r'Z:\Research_Users\David\sample_transient_images-20240724T164104Z-001\sample_transient_images'
data_dirpath = '/Volumes/velten/Research_Users/David/sample_transient_images-20240724T164104Z-001/sample_transient_images'
transient_data_dirpath = r'{}/transient_images_{}x{}_nt-{}'.format(data_dirpath, n_rows, n_cols, n_tbins, )
rgb_data_dirpath = r'{}/rgb_images_{}x{}_nt-{}'.format(data_dirpath, n_rows, n_cols, n_tbins, )
gt_depths_data_dirpath = r'{}/depth_images_{}x{}_nt-{}'.format(data_dirpath,n_rows, n_cols, n_tbins, )


render_params_str = 'nr-{}_nc-{}_nt-{}_samples-{}'.format(n_rows, n_cols, n_tbins, n_samples)

if __name__ == '__main__':


    #scene_id = 'bedroom'
    scene_id = 'living-room-2'

    if scene_id == 'bedroom':
        x1, y1 = (250, 200)
        x2, y2 = (220, 180)
        x3, y3 = (75, 75)
        x4, y4 = (130, 80)
    elif scene_id == 'living-room-2':
        x1, y1 = (100 // factor , 100 // factor)
        x2, y2 = (220 // factor, 180 // factor)
        x3, y3 = (177 // factor, 76 // factor)
        x4, y4 = (178 // factor, 198 // factor)

    scene_filename = r'{}_{}_view-0'.format(scene_id, render_params_str)
    print("Loading: {}".format(scene_filename))
    gt_depths_img = np.load(r'{}/{}.npy'.format(gt_depths_data_dirpath, scene_filename))
    rgb_img = np.load(r'{}/{}.npy'.format(rgb_data_dirpath, scene_filename))
    hist_img = np.load(r'{}/{}.npz'.format(transient_data_dirpath, scene_filename))['arr_0']
    # rgb_image = np.load(os.path.join(rgb_folder, filename))
    # filename = r'C:\Users\Patron\PycharmProjects\Indirect-Direct-ToF\data\horse_depth_map.npy'
    # depth_image = np.load(filename)
    hist_img = np.pad(hist_img[..., ::2] + hist_img[..., 1::2], ((0, 0), (0, 0), (0, 24)))
    #hist_img = np.pad(hist_img, ((0, 0), (0, 0), (0, 48)))

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
    peak_factor = 0.015
    n_codes = [8, 12]
    sigma = 10

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


    for j in range(len(n_codes)):
        n_code = n_codes[j]

        irf = gaussian_pulse(np.arange(params['n_tbins']), 0, sigma, circ_shifted=True)
        if peak_factor is None:
            params['imaging_schemes'] = [
                ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf, constant_pulse_energy=True),
                ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

                ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=n_code, pulse_width=1,
                                    account_irf=True, h_irf=irf),
                ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                     model=os.path.join('bandlimited_models', f'n{n_tbins}_k{n_code}_sigma{sigma}'),
                                    account_irf=True, h_irf=irf),
                ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=n_code, pulse_width=1, account_irf=True, h_irf=irf),
                ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

            ]
        else:
            peak_name = f"{peak_factor:.3f}".split(".")[-1]
            params['imaging_schemes'] = [
                ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf, constant_pulse_energy=True),
                ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf,
                                    constant_pulse_energy=False),

                ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=n_code, pulse_width=1,
                                    account_irf=True, h_irf=irf, constant_pulse_energy=True),
                ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
                                    model=os.path.join('bandlimited_peak_models',
                                                       f'n{params["n_tbins"]}_k{n_code}_sigma{sigma}_peak{peak_name}_counts1000'),
                                    h_irf=irf),
                ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=n_code, pulse_width=1, account_irf=True, h_irf=irf,
                                    constant_pulse_energy=True),
                ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf,
                                    constant_pulse_energy=True),

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

            if imaging_scheme.constant_pulse_energy:
                incident, tmp_irf = light_obj.simulate_constant_pulse_energy(photon_counts, sbr, np.array([0]),
                                                                             peak_factor=peak_factor)
            else:
                incident, tmp_irf = light_obj.simulate_average_photons(photon_counts, sbr, np.array([0]),
                                                                       peak_factor=peak_factor)

            coding_obj.update_tmp_irf(tmp_irf)
            coding_obj.update_C_derived_params()

            incident = np.squeeze(incident)
            hist_img /= np.sum(hist_img, axis=-1, keepdims=True)
            tmp = np.zeros_like(hist_img)
            for r in range(nr):
                for c in range(nc):
                    tmp[r, c, :] = signalproc_ops.circular_conv(hist_img[r, c, :][np.newaxis, ...], incident, axis=-1)

            if coding_scheme == 'Identity' and imaging_scheme.constant_pulse_energy == True:
                coded_vals_tmp = coding_obj.encode_no_noise(np.reshape(tmp, (nr * nc, n_tbins))).squeeze()
                depths_tmp = np.reshape(
                    coding_obj.max_peak_decoding(coded_vals_tmp, rec_algo_id=rec_algo) * tbin_depth_res,
                    (nr, nc))
                # depths_tmp = np.argmax(tmp, axis=-1) * tbin_depth_res
                # band_hist_img = np.reshape(signalproc_ops.circular_corr(np.squeeze(tmp_irf)[np.newaxis, ...],
                #                np.reshape(hist_img, (nr*nc, n_tbins)).copy(), axis=-1), (nr, nc, n_tbins))
                # depths_tmp = np.argmax(tmp, axis=-1) * tbin_depth_res
                error_tmp = np.abs(depths_tmp - gt_depths_img)
                mask = np.logical_not((error_tmp > 0.2))
                gt_tmp = np.copy(gt_depths_img)
                gt_tmp[mask] = depths_tmp[mask]
                depths_const = gt_tmp.flatten()
            elif coding_scheme == 'Identity':
                coded_vals_tmp = coding_obj.encode_no_noise(np.reshape(tmp, (nr * nc, n_tbins))).squeeze()
                depths_tmp = np.reshape(
                    coding_obj.max_peak_decoding(coded_vals_tmp, rec_algo_id=rec_algo) * tbin_depth_res,
                    (nr, nc))
                # depths_tmp = np.argmax(tmp, axis=-1) * tbin_depth_res
                # band_hist_img = np.reshape(signalproc_ops.circular_corr(np.squeeze(tmp_irf)[np.newaxis, ...],
                #                np.reshape(hist_img, (nr*nc, n_tbins)).copy(), axis=-1), (nr, nc, n_tbins))
                # depths_tmp = np.argmax(tmp, axis=-1) * tbin_depth_res
                error_tmp = np.abs(depths_tmp - gt_depths_img)
                mask = np.logical_not((error_tmp > 0.2))
                gt_tmp = np.copy(gt_depths_img)
                gt_tmp[mask] = depths_tmp[mask]
                depths_clip = gt_tmp.flatten()

            coded_vals = coding_obj.encode(np.reshape(tmp, (nr * nc, n_tbins)), 1).squeeze()

            if coding_scheme in ['sIdentity']:
                assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
                decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                                   rec_algo_id=rec_algo) * tbin_depth_res
            else:
                decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

            # decoded_depths = np.argmax(hist_img, axis=-1).flatten() * tbin_depth_res
            if imaging_scheme.constant_pulse_energy:
                depths = depths_const
            else:
                depths = depths_clip

            if k == 0 or k==1:
                continue
            error_maps[:, :, j, k-2] = np.abs(np.reshape(decoded_depths, (nr, nc)) - np.reshape(depths, (nr, nc)))
            depth_images[:, :, j, k-2] = np.reshape(decoded_depths, (nr, nc))
            errors = np.abs(decoded_depths - depths.flatten()) * depth_res
            error_metrix = calc_error_metrics(errors)
            rmse[j, k-2] = error_metrix['rmse']
            mae[j, k-2] = error_metrix['mae']

    #errors_maps = error_maps[:, :, :, 1:]
    #depth_images = depth_images[:, :, :, 1:]

    n_codes_size = len(n_codes)
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(n_codes_size + 1, schemes_num+1, figure=fig, hspace=0.05, wspace=0.05)

    for j in range(n_codes_size):
        for k in range(schemes_num):

            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[j, k],
                                                        width_ratios=[1, 1], hspace=0, wspace=0.05)

            ax_low = fig.add_subplot(inner_gs[0, 0])
            ax_high = fig.add_subplot(inner_gs[0, 1])

            depth_im = ax_low.imshow(depth_images[:,:, j, k], vmin=depths.min(), vmax=depths.max())
            error_im = ax_high.imshow(error_maps[:, :, j, k] * 100, vmin=0, vmax=26)

            ax_low.text(
                2, 2,  # x, y pixel coordinates
                f"MAE: {mae[j, k] / 10:.2f} cm \nRMSE {rmse[j, k] / 10: .2f} cm",  # your text
                color='white',
                fontsize=8,
                ha='left', va='top',  # align text relative to point
                bbox=dict(facecolor='black', alpha=0.5, pad=2)  # optional background box
            )
            ax_low.get_xaxis().set_ticks([])
            ax_low.get_yaxis().set_ticks([])
            ax_high.get_xaxis().set_ticks([])
            ax_high.get_yaxis().set_ticks([])
            if j == n_codes_size - 1:
                inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[j + 1, k],
                                                            width_ratios=[1, 1], hspace=0, wspace=0.05)

                ax_low = fig.add_subplot(inner_gs[0, 0])
                ax_high = fig.add_subplot(inner_gs[0, 1])

                ax_low.get_xaxis().set_ticks([])
                ax_low.get_yaxis().set_ticks([])
                ax_high.get_xaxis().set_ticks([])
                ax_high.get_yaxis().set_ticks([])

                cbar_im = fig.colorbar(depth_im, ax=ax_low, orientation='horizontal', label='Depth (meters)', pad=0.04)
                cbar_error = fig.colorbar(error_im, ax=ax_high, orientation='horizontal', label='Error (cm)',
                                          pad=0.04)

                #cbar_im = fig.colorbar(depth_im, ax=ax_low, orientation='horizontal', label='Depth (meters)', pad=0.04)
                #cbar_error = fig.colorbar(error_im, ax=ax_high, orientation='horizontal', label='Error (cm)', pad=0.04)



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

        if j == 2:
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[j, k + 1], width_ratios=[1, 1], hspace=0,
                                                        wspace=0.05)

            ax_rgb = fig.add_subplot(inner_gs[0, 0])
            ax_rgb.imshow(gt_depths_img)
            ax_rgb.plot(x4, y4, 'o', color='blue', markersize=6)
            ax_rgb.plot(x2, y2, 'o', color='orange', markersize=6)
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
            ax_low.plot(x4, y4, 'o', color='blue', markersize=6)
            ax_low.plot(x2, y2, 'o', color='orange', markersize=6)
            error_im = ax_high.imshow(error_maps[:, :, j, k+1] * 100, vmin=0, vmax=26)

            ax_low.get_xaxis().set_ticks([])
            ax_low.get_yaxis().set_ticks([])
            ax_high.get_xaxis().set_ticks([])
            ax_high.get_yaxis().set_ticks([])

            #cbar_im = fig.colorbar(depth_im, ax=ax_low, orientation='horizontal', label='Depth (meters)', pad=0.04)
            #cbar_error = fig.colorbar(error_im, ax=ax_high, orientation='horizontal', label='Error (cm) and RGB', pad=0.04)

            ax_low.text(
                2, 2,  # x, y pixel coordinates
                f"MAE: {mae[j, k+1] / 10:.2f} cm \nRMSE {rmse[j, k+1] / 10: .2f} cm",  # your text
                color='white',
                fontsize=8,
                ha='left', va='top',  # align text relative to point
                bbox=dict(facecolor='black', alpha=0.5, pad=2)  # optional background box
            )
        elif j == 0:
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=gs[j, k + 1], width_ratios=[1, 1], hspace=0, wspace=0.05
            )

            ax_low = fig.add_subplot(inner_gs[0, 0])
            ax_high = fig.add_subplot(inner_gs[0, 1])

            # Plot the histograms
            ax_low.plot(hist_img[y4, x4], color='blue')
            #ax_low.plot(hist_img[210, 220], color='orange')

            #ax_low.set_ylim(-0.00001, 0.0003)
            ax_high.plot(hist_img[y2, x2], color='orange')
            #ax_high.set_ylim(-0.00001, 0.0003)

            #ax_low.set_xticks((np.linspace(0, 2000, 3)))
            #ax_low.set_xticklabels((np.linspace(0, 2000, 3) * tbin_res * 1e9).astype(int))
            #ax_low.tick_params(axis='both', which='major', labelsize=6)
            ax_low.set_yticks([])
            ax_low.set_xticks([])

            ax_low.set_xlabel('Time')
            #ax_low.set_ylabel('Counts')
            ax_low.set_box_aspect(1 / 2)  # match 1 row : 2 cols like your images


            #ax_high.set_xticks((np.linspace(0, 2000, 3)))
            #ax_high.set_xticklabels((np.linspace(0, 2000, 3) * tbin_res * 1e9).astype(int))
            #ax_high.tick_params(axis='both', which='major', labelsize=6)
            ax_high.set_yticks([])
            ax_high.set_xticks([])

            ax_high.set_xlabel('Time')
            ax_high.set_box_aspect(1 / 2)  # match 1 row : 2 cols like your images

    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[j + 1, k+1],
                                                    width_ratios=[1, 1], hspace=0, wspace=0.05)

    ax_low = fig.add_subplot(inner_gs[0, 0])
    ax_high = fig.add_subplot(inner_gs[0, 1])

    ax_low.get_xaxis().set_ticks([])
    ax_low.get_yaxis().set_ticks([])
    ax_high.get_xaxis().set_ticks([])
    ax_high.get_yaxis().set_ticks([])

    cbar_im = fig.colorbar(depth_im, ax=ax_low, orientation='horizontal', label='Depth (meters)', pad=0.04)
    cbar_error = fig.colorbar(error_im, ax=ax_high, orientation='horizontal', label='Error (cm)',
                                  pad=0.04)

        # cbar_im = fig.colorbar(depth_im, ax=ax_low, orientation='horizontal', label='Depth (meters)', pad=0.04)
        # cbar_error = fig.colorbar(error_im, ax=ax_high, orientation='horizontal', label='Error (cm)', pad=0.04)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(f'bit_depth_grid_band.svg', bbox_inches='tight', dpi=1000)
    plt.show(block=True)

print('YAYYY')
