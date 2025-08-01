# Python imports
# Library imports
import time
import os

from IPython.core import debugger

from plot_figures.plot_utils import get_scheme_color
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils.research_utils.np_utils import calc_error_metrics
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from felipe_utils import tof_utils_felipe
import felipe_utils.research_utils.signalproc_ops as signalproc_ops
import numpy as np
from matplotlib import rc
import glob
import matplotlib.gridspec as gridspec

font = {'family': 'serif',
        'weight': 'bold',
        'size': 24}

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
factor = 2
n_rows = 240 // factor
n_cols = 320 // factor
n_tbins = 2000
n_samples = 2048

#data_dirpath = 'Z:\Research_Users\David\sample_transient_images-20240724T164104Z-001\sample_transient_images'
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
    constant_pulse_energy = False
    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 10, circ_shifted=True)
    params['imaging_schemes'] = [
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf,
                            constant_pulse_energy=True),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf,
                            constant_pulse_energy=False),

        # ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=32),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=12, pulse_width=1, account_irf=True, h_irf=irf,
                            constant_pulse_energy=constant_pulse_energy),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_models', 'n2000_k8_sigma30'),
        #                     pulse_width=1, account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_models', 'n2000_k10_sigma30'),
        #                    pulse_width=1, account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_models', 'n2000_k12_sigma30'),
        #                     pulse_width=1, account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_models', 'n2000_k12_sigma30_v2'),
        #                     pulse_width=1, account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_models', 'n2000_k14_sigma30'),
        #                     pulse_width=1, account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_models', 'version_6'),
        #                     pulse_width=1, account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                      model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak015_counts1000'),
        #                      account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k10_sigma10_peak015_counts1000'),
        #                     account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                            model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak030_counts1000'),
                            account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                            model=os.path.join('bandlimited_peak_models', 'n1024_k12_sigma5_peak030_counts1000'),
                            account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                            model=os.path.join('bandlimited_peak_models', 'n1024_k12_sigma10_peak030_counts1000'),
                            account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k10_sigma10_peak005_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k12_sigma10_peak005_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k10_sigma5_peak030_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k12_sigma5_peak030_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k10_sigma10_peak015_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k12_sigma10_peak015_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k14_sigma10_peak015_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_models', 'n1024_k12_sigma30'),
        #                     pulse_width=1, account_irf=True, h_irf=irf),
        ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf,
                            constant_pulse_energy=constant_pulse_energy),
        #

    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()


    # Do either average photon count
    photon_counts = 1000
    sbr = 0.1
    peak_factor = 0.030

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

    for i in range(len(imaging_schemes)):
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        #incident = np.zeros((nr * nc, n_tbins))
        #tmp = np.reshape(hist_img, (nr * nc, n_tbins)).copy()

        if imaging_scheme.constant_pulse_energy:
            incident, tmp_irf = light_obj.simulate_constant_pulse_energy(photon_counts, sbr, np.array([0]), peak_factor=peak_factor)
        else:
            incident, tmp_irf = light_obj.simulate_average_photons(photon_counts, sbr, np.array([0]), peak_factor=peak_factor)


        print(f'photon counts: {np.sum(incident[0, 0,: ])}')
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
            depths_tmp = np.reshape(coding_obj.max_peak_decoding(coded_vals_tmp, rec_algo_id=rec_algo) * tbin_depth_res,
                                   (nr, nc))
            #depths_tmp = np.argmax(tmp, axis=-1) * tbin_depth_res
            #band_hist_img = np.reshape(signalproc_ops.circular_corr(np.squeeze(tmp_irf)[np.newaxis, ...],
            #                np.reshape(hist_img, (nr*nc, n_tbins)).copy(), axis=-1), (nr, nc, n_tbins))
            #depths_tmp = np.argmax(tmp, axis=-1) * tbin_depth_res
            error_tmp = np.abs(depths_tmp - gt_depths_img)
            mask = np.logical_not((error_tmp > 0.2))
            gt_tmp = np.copy(gt_depths_img)
            gt_tmp[mask] = depths_tmp[mask]
            depths_const = gt_tmp.flatten()
        elif coding_scheme == 'Identity' and imaging_scheme.constant_pulse_energy == False:
            coded_vals_tmp = coding_obj.encode_no_noise(np.reshape(tmp, (nr * nc, n_tbins))).squeeze()
            depths_tmp = np.reshape(coding_obj.max_peak_decoding(coded_vals_tmp, rec_algo_id=rec_algo) * tbin_depth_res,
                                   (nr, nc))
            #depths_tmp = np.argmax(tmp, axis=-1) * tbin_depth_res
            #band_hist_img = np.reshape(signalproc_ops.circular_corr(np.squeeze(tmp_irf)[np.newaxis, ...],
            #                np.reshape(hist_img, (nr*nc, n_tbins)).copy(), axis=-1), (nr, nc, n_tbins))
            #depths_tmp = np.argmax(tmp, axis=-1) * tbin_depth_res
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

        #decoded_depths = np.argmax(hist_img, axis=-1).flatten() * tbin_depth_res
        if imaging_scheme.constant_pulse_energy:
            depths = depths_const
        else:
            depths = depths_clip

        error_maps[:, :, i] = np.abs(np.reshape(decoded_depths, (nr, nc)) - np.reshape(depths, (nr, nc)))
        depth_images[:, :, i] = np.reshape(decoded_depths, (nr, nc))
        errors = np.abs(decoded_depths - depths.flatten()) * depth_res
        error_metrix = calc_error_metrics(errors)
        rmse[i] = error_metrix['rmse']
        mae[i] = error_metrix['mae']

        if 'Learned' in imaging_schemes[i].coding_id:
            tmp_learned = tmp[y1, x1, :]
            tmp2_learned = tmp[y2, x2, :]
            tmp3_learned = tmp[y3, x3, :]
            tmp4_learned = tmp[y4, x4, :]
        elif 'Identity' in imaging_schemes[i].coding_id:
            tmp1 = tmp[y1, x1, :]
            tmp2 = tmp[y2, x2, :]
            tmp3 = tmp[y3, x3, :]
            tmp4 = tmp[y4, x4, :]

    toc = time.perf_counter()

    n_rows = 3  # 2 main rows + 1 short bottom row
    n_cols = len(params['imaging_schemes']) + 1

    fig = plt.figure(figsize=(20.5, 8))
    gs = gridspec.GridSpec(nrows=n_rows, ncols=n_cols, height_ratios=[1, 1, 0.6])  # Shorter last row
    axs = np.empty((n_rows, n_cols), dtype=object)

    # Create subplots manually
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i, j] = fig.add_subplot(gs[i, j])

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    counter1 = 0
    counter2 = 1

    for i in range(len(params['imaging_schemes'])):

        scheme = params['imaging_schemes'][i]
        depth_map = depth_images[:, :, i]

        #depth_map[depth_map < 1/2*np.min(depth_image)] = np.nan
        #depth_map[depth_map > 2*np.max(depth_image)] = np.nan

        error_map = error_maps[:, :, i] * 100

        depth_map[np.isnan(depth_map)] = 0

        if scheme.coding_id == 'Identity':
            depth_im = axs[counter1][i].imshow(rgb_img, vmax=depths.max(), vmin=depths.min())
            axs[counter1][i].plot(x1, y1, 'o', color='purple', markersize=10)
            axs[counter1][i].plot(x2, y2, 'o', color='green', markersize=10)
            axs[counter1][i].plot(x3, y3, 'o', color='blue', markersize=10)
            axs[counter1][i].plot(x4, y4, 'o', color='red', markersize=10)
        else:
            depth_im = axs[counter1][i].imshow(depth_map, vmax=depths.max(), vmin=depths.min())

        for spine in axs[counter1][i].spines.values():
            spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
            spine.set_linewidth(2)

        error_im = axs[counter2][i].imshow(error_map, vmin=0, vmax=6)

        for spine in axs[counter2][i].spines.values():
            spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
            spine.set_linewidth(2)

        axs[counter1][i].get_xaxis().set_ticks([])
        axs[counter1][i].get_yaxis().set_ticks([])
        axs[counter2][i].get_xaxis().set_ticks([])
        axs[counter2][i].get_yaxis().set_ticks([])
        #if counter == 2:
        axs[counter1][i].text(
                2, 2,  # x, y pixel coordinates
                f"RMSE:{rmse[i] / 10:.1f} cm\nMAE:{mae[i] / 10:.1f} cm",  # your text
                color='white',
                fontsize=20,
                ha='left', va='top',  # align text relative to point
                bbox=dict(facecolor='black', alpha=0.5, pad=1)  # optional background box
            )

        if scheme.coding_id.startswith('TruncatedFourier'):
            str_name = 'Trunc. Fourier'
        elif scheme.coding_id == 'Identity':
            str_name = 'RGB & FRH'
        elif scheme.coding_id == 'Greys':
            str_name = 'Count. Gray'
        elif scheme.coding_id.startswith('Learned'):
            str_name = 'Optimized'
        axs[0][i].set_title(str_name)
        print(f'Scheme: {scheme.coding_id}, RMSE: {rmse[i] / 10: .2f} cm, MAE: {mae[i] / 10:.2f} cm')


    # axs[0][-1].plot(tmp_learned, color='crimson', label='Non-pulse')
    # axs[1][-1].plot(tmp2_learned, color='green', label='Non-pulse')
    # axs[2][-1].plot(tmp3_learned, color='blue', label='Non-pulse')
    # axs[3][-1].plot(tmp4_learned, color='red', label='Non-pulse')

    if n_tbins == 2000:
        axs[-1][0].plot(tmp1[500:1700], color='purple', label='Pulse', linewidth=2.5)
        axs[-1][1].plot(tmp2[500:1700], color='green', label='Pulse', linewidth=2.5)
        axs[-1][2].plot(tmp3[500:1700], color='blue', label='Pulse', linewidth=2.5)
        axs[-1][3].plot(tmp4[500:1700], color='red', label='Pulse', linewidth=2.5)

        xticks = np.linspace(0, 1200, 5)
        xtick_labels = (np.linspace(500, 1650, 5) * tbin_res * 1e9).astype(int)
    else:
        axs[-1][0].plot(tmp1[100:1000], color='purple', label='Pulse', linewidth=2.5)
        axs[-1][1].plot(tmp2[100:1000], color='green', label='Pulse', linewidth=2.5)
        axs[-1][2].plot(tmp3[100:1000], color='blue', label='Pulse', linewidth=2.5)
        axs[-1][3].plot(tmp4[100:1000], color='red', label='Pulse', linewidth=2.5)

        xticks = np.linspace(0, 900, 5)
        xtick_labels = (np.linspace(100, 1050, 5) * tbin_res * 1e9).astype(int)
    for i in range(4):
        axs[-1][i].set_yticks([])
        axs[-1][i].set_xticks(xticks)
        axs[-1][i].set_xticklabels(xtick_labels, fontsize=20)  # <-- Set desired font size here

    axs[0, -1].axis('off')
    axs[1, -1].axis('off')
    axs[2, -1].axis('off')
    cbar_im = fig.colorbar(depth_im, ax=axs[0, :], orientation='vertical')
    cbar_error = fig.colorbar(error_im, ax=axs[1, :], orientation='vertical')
    cbar_im.ax.tick_params(labelsize=20)
    cbar_error.ax.tick_params(labelsize=20)

    axs[0, -1].legend()
    axs[1, -1].legend()
    #fig.tight_layout()
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    #fig.savefig('tmp2.svg', bbox_inches='tight', dpi=3000)
    plt.show(block=True)
    print()
print('YAYYY')
