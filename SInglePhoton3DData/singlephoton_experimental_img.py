from felipe_utils.research_utils.io_ops import load_json
from felipe_utils.scan_data_utils import *
from felipe_utils import tof_utils_felipe
import os
from scipy.ndimage import gaussian_filter
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from spad_toflib.coding_schemes import IdentityCoding
import matplotlib.pyplot as plt
from plot_figures.plot_utils import get_scheme_color

global_shift = 0
downsamp_factor = 1  # Spatial downsample factor
hist_tbin_factor = 1.0  # increase tbin size to make histogramming faster

def get_ground_truth(scene_id):
    scan_data_params = load_json('scan_params.json')
    io_dirpaths = load_json('io_dirpaths.json')
    hist_img_base_dirpath = io_dirpaths["preprocessed_hist_data_base_dirpath"]
    assert (scene_id in scan_data_params['scene_ids']), "{} not in scene_ids".format(scene_id)
    hist_dirpath = os.path.join(hist_img_base_dirpath, scene_id)

    ## Histogram image params

    n_rows_fullres = scan_data_params['scene_params'][scene_id]['n_rows_fullres']
    n_cols_fullres = scan_data_params['scene_params'][scene_id]['n_cols_fullres']
    (nr, nc) = (n_rows_fullres // downsamp_factor, n_cols_fullres // downsamp_factor)  # dims for face_scanning scene
    min_tbin_size = scan_data_params['min_tbin_size']  # Bin size in ps
    hist_tbin_size = min_tbin_size * hist_tbin_factor  # increase size of time bin to make histogramming faster
    hist_img_tau = scan_data_params['hist_preprocessing_params']['hist_end_time'] - \
                   scan_data_params['hist_preprocessing_params']['hist_start_time']
    nt = get_nt(hist_img_tau, hist_tbin_size)

    ## Load histogram image
    hist_img_fname = get_hist_img_fname(nr, nc, hist_tbin_size, hist_img_tau, is_unimodal=False)
    hist_img_fpath = os.path.join(hist_dirpath, hist_img_fname)
    hist_img = np.load(hist_img_fpath)

    ## Shift histogram image if needed
    hist_img = np.roll(hist_img, global_shift, axis=-1)

    irf = get_scene_irf(scene_id, nt, tlen=hist_img_tau, is_unimodal=False)

    coding_obj = IdentityCoding(n_tbins=hist_img.shape[-1], account_irf=True, h_irf=irf)
    matchfilt_tof = coding_obj.max_peak_decoding(hist_img, rec_algo_id='matchfilt').squeeze() * time2depth(hist_tbin_size * 1e-12)
    return matchfilt_tof


if __name__=='__main__':

    ## Load parameters shared by all
    scan_data_params = load_json('scan_params.json')
    io_dirpaths = load_json('io_dirpaths.json')
    hist_img_base_dirpath = io_dirpaths["preprocessed_hist_data_base_dirpath"]

    ## Load processed scene:
    scene_id = '20190209_deer_high_mu/free'
    #scene_id = '20190207_face_scanning_low_mu/free'
    #scene_id = '20190207_face_scanning_low_mu/ground_truth'

    #scene_id = '20181105_face/opt_flux'

    #scene_id = '20190207_face_scanning_low_mu/ground_truth'
    #depths = get_ground_truth('20190207_face_scanning_low_mu/ground_truth')
    depths = get_ground_truth(scene_id)


    assert (scene_id in scan_data_params['scene_ids']), "{} not in scene_ids".format(scene_id)
    hist_dirpath = os.path.join(hist_img_base_dirpath, scene_id)

    ## Histogram image params
    n_rows_fullres = scan_data_params['scene_params'][scene_id]['n_rows_fullres']
    n_cols_fullres = scan_data_params['scene_params'][scene_id]['n_cols_fullres']
    (nr, nc) = (n_rows_fullres // downsamp_factor, n_cols_fullres // downsamp_factor)  # dims for face_scanning scene
    min_tbin_size = scan_data_params['min_tbin_size']  # Bin size in ps
    hist_tbin_size = min_tbin_size * hist_tbin_factor  # increase size of time bin to make histogramming faster
    hist_img_tau = scan_data_params['hist_preprocessing_params']['hist_end_time'] - \
                   scan_data_params['hist_preprocessing_params']['hist_start_time']
    nt = get_nt(hist_img_tau, hist_tbin_size)

    ## Load histogram image
    hist_img_fname = get_hist_img_fname(nr, nc, hist_tbin_size, hist_img_tau, is_unimodal=False)
    hist_img_fpath = os.path.join(hist_dirpath, hist_img_fname)
    hist_img = np.load(hist_img_fpath)

    ## Shift histogram image if needed
    hist_img = np.roll(hist_img, global_shift, axis=-1)

    denoised_hist_img = gaussian_filter(hist_img, sigma=0.75, mode='wrap', truncate=1)
    (tbins, tbin_edges) = get_hist_bins(hist_img_tau, hist_tbin_size)

    irf_tres = scan_data_params['min_tbin_size']  # in picosecs
    irf = get_scene_irf(scene_id, nt, tlen=hist_img_tau, is_unimodal=False)

    params = {}
    params['n_tbins'] = tbins.shape[-1]
    #params['dMax'] = 5
    #params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = scan_data_params['laser_rep_freq']
    params['rep_tau'] = 1. / params['rep_freq']
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['T'] = 0.1 #intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM

    params['imaging_schemes'] = list(reversed([
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=10, pulse_width=1,  account_irf=False,
                            h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'version_1'),
        #                     account_irf=True, h_irf=irf),
        ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=10, pulse_width=1, account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'n2188_k8_spaddata_v2'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'n2188_k8_spaddata'),
        #                     account_irf=True, h_irf=irf),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                            model=os.path.join('bandlimited_models', 'version_4_v2'),
                            account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'version_4_v3'),
        #                     account_irf=True, h_irf=irf),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

    ]))


    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]


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

    init_coding_list(n_tbins, None, params, t_domain=t_domain)
    imaging_schemes = params['imaging_schemes']

    depth_images = np.zeros((nr, nc, len(params['imaging_schemes'])))
    error_maps = np.zeros((nr, nc, len(params['imaging_schemes'])))
    byte_sizes = np.zeros((len(params['imaging_schemes'])))
    rmse = np.zeros((len(params['imaging_schemes'])))
    mae = np.zeros((len(params['imaging_schemes'])))


    for i in range(len(imaging_schemes)):
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        rec_algo = imaging_scheme.rec_algo

        if coding_scheme == 'Identity':
            coded_vals = hist_img.reshape(nr * nc, params['n_tbins'])
        else:
            coded_vals = coding_obj.encode_no_noise(hist_img.reshape(nr * nc, params['n_tbins'])).squeeze()


        decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * time2depth(hist_tbin_size * 1e-12)

        if 'face_scanning' in scene_id:
            mask = plt.imread(io_dirpaths['hist_mask_path'])
            mask = np.logical_not(mask)
            depths = mask * depths
            decoded_depths = mask.flatten() * decoded_depths
        elif 'deer' in scene_id:
            mask = plt.imread(r'C:\Users\Patron\PycharmProjects\Indirect-Direct-ToF\SInglePhoton3DData\deer.png')[..., 0]
            mask[mask > 0] = 1
            depths = mask * depths
            decoded_depths = mask.flatten() * decoded_depths

        normalized_decoded_depths = np.copy(decoded_depths)
        normalized_decoded_depths[normalized_decoded_depths == 0] = np.nan
        # vmin = np.nanmean(normalized_decoded_depths) - 1
        # vmax = np.nanmean(normalized_decoded_depths) + 1

        #porcleain_face
        # vmin = 0.04
        # vmax = 1.0

        #dear
        vmin = 0.05
        vmax = 1.2
        normalized_decoded_depths[normalized_decoded_depths > vmax] = np.nan
        normalized_decoded_depths[normalized_decoded_depths < vmin] = np.nan
        # normalized_decoded_depths = (normalized_decoded_depths - np.nanmean(normalized_decoded_depths)) / np.nanstd(normalized_decoded_depths)
        error_maps[:, :, i] = np.abs(np.reshape(decoded_depths, (nr, nc)) - depths)
        depth_images[:, :, i] = np.reshape(normalized_decoded_depths, (nr, nc))
        byte_sizes[i] = np.squeeze(coded_vals).size * np.squeeze(coded_vals).itemsize
        errors = np.abs(decoded_depths - depths.flatten()[np.newaxis, :]) * depth_res
        error_metrix = calc_error_metrics(errors[~np.isnan(errors)])


        rmse[i] = error_metrix['rmse']
        mae[i] = error_metrix['mae']


    fig, axs = plt.subplots(2, len(params['imaging_schemes'])+1, squeeze=False, figsize=(10, 5))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)


    axs[0][0].set_ylabel('Depth Map')
    axs[1][0].set_ylabel('Depth Errors (mm)')

    for i in range(len(params['imaging_schemes'])):

        scheme = params['imaging_schemes'][i]
        depth_map = depth_images[:, :, i]

        # depth_map[depth_map < 1/2*np.min(depth_image)] = np.nan
        # depth_map[depth_map > 2*np.max(depth_image)] = np.nan

        error_map = error_maps[:, :, i] * 10

        depth_im = axs[0][i].imshow(depth_map,
                                           vmin=np.nanmin(depth_images), vmax=np.nanmax(depth_images))
        for spine in axs[0][i].spines.values():
            spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
            spine.set_linewidth(2)

        error_im = axs[1][i].imshow(error_map, vmin=0, vmax=2)

        for spine in axs[1][i].spines.values():
            spine.set_edgecolor(get_scheme_color(scheme.coding_id, k=scheme.coding_obj.n_functions))  # Set border color
            spine.set_linewidth(2)

        axs[0][i].get_xaxis().set_ticks([])
        axs[0][i].get_yaxis().set_ticks([])
        axs[1][i].get_xaxis().set_ticks([])
        axs[1][i].get_yaxis().set_ticks([])
        #if counter == 2:
        axs[1][i].set_xlabel(f'RMSE:{rmse[i]/10:.2f} cm \n MAE:{mae[i] / 10:.2f} cm')

        str_name = ''
        if imaging_schemes[i].coding_id.startswith('TruncatedFourier'):
            str_name = 'Truncated Fourier'
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
            str_name = 'Optimized'

        axs[0][i].set_title(str_name)
        print(f'Scheme: {scheme.coding_id}, RMSE: {rmse[i] / 10: .2f} cm, MAE: {mae[i] / 10:.2f} cm')

    axs[0, -1].axis('off')
    axs[1, -1].axis('off')
    cbar_im = fig.colorbar(depth_im, ax=axs[0, -1], orientation='vertical', label='Depth (cm)')
    cbar_error = fig.colorbar(error_im, ax=axs[1, -1], orientation='vertical', label='Error (cm)')

    axs[0, -1].legend()
    axs[1, -1].legend()
    #fig.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.savefig(f'Z:\\Research_Users\\David\\Learned Coding Functions Paper\\supp_exp.svg', bbox_inches='tight')
    plt.show()
    print(scan_data_params)

