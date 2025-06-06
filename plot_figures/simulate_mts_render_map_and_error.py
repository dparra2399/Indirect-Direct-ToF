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

n_rows = 240
n_cols = 320
n_tbins = 2000
n_samples = 2048

data_dirpath = r'Z:\Research_Users\David\sample_transient_images-20240724T164104Z-001\sample_transient_images'
transient_data_dirpath = r'{}\transient_images_{}x{}_nt-{}'.format(data_dirpath, n_rows, n_cols, n_tbins, )
rgb_data_dirpath = r'{}\rgb_images_{}x{}_nt-{}'.format(data_dirpath, n_rows, n_cols, n_tbins, )
gt_depths_data_dirpath = r'{}\depth_images_{}x{}_nt-{}'.format(data_dirpath,n_rows, n_cols, n_tbins, )



render_params_str = 'nr-{}_nc-{}_nt-{}_samples-{}'.format(n_rows, n_cols, n_tbins, n_samples)

if __name__ == '__main__':


    scene_id = 'bedroom'
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

    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 30, circ_shifted=True)
    params['imaging_schemes'] = [
        # ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=32),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=16, pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Gaussian', 'zncc', model=os.path.join('bandlimited_models', 'n2000_k8_sigma30'),
                             pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Gaussian', 'zncc', model=os.path.join('bandlimited_models', 'n2000_k12_sigma30'),
                            pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Gaussian', 'zncc',
                            model=os.path.join('bandlimited_models', 'version_3'),
                            pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Gaussian', 'zncc',
                            model=os.path.join('bandlimited_models', 'version_4'),
                            pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Gaussian', 'zncc',
                            model=os.path.join('bandlimited_models', 'version_5'),
                            pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Gaussian', 'zncc',
                            model=os.path.join('bandlimited_models', 'version_6'),
                            pulse_width=1, account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Gaussian', 'zncc', model=os.path.join('bandlimited_models', 'n2000_k10_sigma30'),
        #                     pulse_width=1, account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak030_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_models', 'n1024_k12_sigma30'),
        #                     pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=10, pulse_width=1, account_irf=True, h_irf=irf),
        #
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),


    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()


    # Do either average photon count
    photon_counts = 1000
    sbr = 0.1
    peak_factor = None

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

    init_coding_list(n_tbins, params, t_domain=t_domain)
    imaging_schemes = params['imaging_schemes']
    depth_images = np.zeros((nr, nc, len(params['imaging_schemes'])))
    error_maps = np.zeros((nr, nc, len(params['imaging_schemes'])))

    rmse = np.zeros((len(params['imaging_schemes'])))
    mae = np.zeros((len(params['imaging_schemes'])))

    tic = time.perf_counter()

    hist_img = np.roll(hist_img, 12, axis=-1)
    #depths = np.argmax(hist_img, axis=-1).flatten() * tbin_depth_res

    for i in range(len(imaging_schemes)):
        imaging_scheme = imaging_schemes[i]
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
        for j in range(nr * nc):
            first = tmp[j, :] * (photon_counts / np.sum(tmp[j, :]).astype(np.float64))
            tmp_irf = tmp_irf * (photon_counts / np.sum(tmp_irf).astype(np.float64))
            incident[j, :] = first + (total_amb_photons / n_tbins)
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

        tmp2 = np.reshape(incident, (nr, nc, n_tbins))



        #coded_vals = coding_obj.encode(incident, 1).squeeze()
        coded_vals = coding_obj.encode_no_noise(incident).squeeze()


        if coding_scheme in ['sIdentity']:
            assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
            decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                               rec_algo_id=rec_algo) * tbin_depth_res
        else:
            decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res


        error_maps[:, :, i] = np.abs(np.reshape(decoded_depths, (nr, nc)) - np.reshape(depths, (nr, nc)))
        depth_images[:, :, i] = np.reshape(decoded_depths, (nr, nc))
        errors = np.abs(decoded_depths - depths.flatten()) * depth_res
        error_metrix = calc_error_metrics(errors)
        rmse[i] = error_metrix['rmse']
        mae[i] = error_metrix['mae']

    toc = time.perf_counter()

    fig, axs = plt.subplots(2, len(params['imaging_schemes'])+1, squeeze=False, figsize=(16, 5))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    counter1 = 0
    counter2 = 1

    axs[0][0].imshow(np.reshape(depths, (nr, nc)), vmax=depths.max(), vmin=depths.min())
    axs[1][0].imshow(rgb_img)


    for spine in axs[0][0].spines.values():
        spine.set_edgecolor('black')  # Set border color
        spine.set_linewidth(2)

    for spine in axs[1][0].spines.values():
        spine.set_edgecolor('black')  # Set border color
        spine.set_linewidth(2)




    axs[0][0].get_xaxis().set_ticks([])
    axs[0][0].get_yaxis().set_ticks([])
    axs[1][0].get_xaxis().set_ticks([])
    axs[1][0].get_yaxis().set_ticks([])
    axs[0][0].set_title('Ground Truth')
    axs[1][0].set_xlabel('RGB Image')


    for i in range(len(params['imaging_schemes'])):

        scheme = params['imaging_schemes'][i]
        depth_map = depth_images[:, :, i]

        #depth_map[depth_map < 1/2*np.min(depth_image)] = np.nan
        #depth_map[depth_map > 2*np.max(depth_image)] = np.nan

        error_map = error_maps[:, :, i] * 10

        depth_map[np.isnan(depth_map)] = 0
        depth_im = axs[counter1][i+1].imshow(depth_map, vmax=depths.max(), vmin=depths.min())

        for spine in axs[counter1][i+1].spines.values():
            spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
            spine.set_linewidth(4)

        error_im = axs[counter2][i+1].imshow(error_map, vmin=0, vmax=5)

        for spine in axs[counter2][i+1].spines.values():
            spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
            spine.set_linewidth(4)

        axs[counter1][i+1].get_xaxis().set_ticks([])
        axs[counter1][i+1].get_yaxis().set_ticks([])
        axs[counter2][i+1].get_xaxis().set_ticks([])
        axs[counter2][i+1].get_yaxis().set_ticks([])
        #if counter == 2:
        axs[counter2][i+1].set_xlabel(f'RMSE: {rmse[i] / 10: .2f} cm \n MAE: {mae[i] / 10:.2f} cm')
        axs[0][i+1].set_title(get_string_name(scheme))
        print(f'Scheme: {scheme.coding_id}, RMSE: {rmse[i] / 10: .2f} cm, MAE: {mae[i] / 10:.2f} cm')


    axs[0, -1].axis('off')
    axs[1, -1].axis('off')
    cbar_im = fig.colorbar(depth_im, ax=axs[0, :], orientation='vertical', label='Depth (meters)')
    cbar_error = fig.colorbar(error_im, ax=axs[1, :], orientation='vertical', label='Error (meters)')

    axs[0, -1].legend()
    axs[1, -1].legend()
    #fig.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    #fig.savefig('Z:\\Research_Users\\David\\Learned Coding Functions Paper\\supp_k4_sigma20_high.svg', bbox_inches='tight', dpi=3000)
    plt.show(block=True)
    print()
print('YAYYY')
