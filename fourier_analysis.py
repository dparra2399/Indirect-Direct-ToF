import numpy as np
import matplotlib.pyplot as plt
from felipe_utils.tof_utils_felipe import zero_norm_t

import numpy as np
from IPython.core import debugger
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from plot_figures.plot_utils import *
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

font = {'family': 'serif',
        'size': 26
        }
matplotlib.rc('font', **font)

def reconstruct_and_get_code_global_top_n(coding_matrix, num_coefficients_to_keep):
    """
    Performs Fourier transform on the entire coding matrix and keeps only the
    largest N magnitude coefficients across all columns for reconstruction.

    Args:
        coding_matrix (numpy.ndarray): A 2D numpy array representing the coding matrix (rows, cols).
        num_coefficients_to_keep (int): The total number of largest magnitude coefficients to keep across the entire matrix.

    Returns:
        numpy.ndarray: The reconstructed coding matrix.
    """
    fourier_coeffs = np.fft.fft(coding_matrix, axis=0)
    magnitudes = np.abs(fourier_coeffs)

    # Flatten the magnitude array to find the top N indices globally
    magnitudes_flat = magnitudes.flatten()
    ind_sorted_flat = np.argsort(magnitudes_flat)[::-1]
    top_n_indices_flat = ind_sorted_flat[:num_coefficients_to_keep]

    # Create a mask for the Fourier coefficients
    mask = np.zeros_like(fourier_coeffs, dtype=bool)
    rows, cols = coding_matrix.shape
    row_indices, col_indices = np.unravel_index(top_n_indices_flat, (rows, cols))

    # Set the mask to True for the top N coefficients
    mask[row_indices, col_indices] = True

    # Apply the mask to zero out the smaller coefficients
    modified_fourier_coeffs = np.where(mask, fourier_coeffs, 0)

    # Perform inverse Fourier transform
    reconstructed_matrix = np.fft.ifft(modified_fourier_coeffs, axis=0).real
    return reconstructed_matrix



# Example Usage:
if __name__ == "__main__":
    # Create an example coding matrix


    params = {}
    params['n_tbins'] = 1024
    #params['dMax'] = 5
    #params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 5 * 1e6
    params['rep_tau'] = 1. / params['rep_freq']
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['T'] = 0.1 #intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM

    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]

    params['imaging_schemes'] = [
        # ImagingSystemParams('Greys', 'Gaussian', 'ncc', pulse_width=1, n_bits=8,
        #                     account_irf=True,
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 1, circ_shifted=True)),
        #
        # ImagingSystemParams('Greys', 'Gaussian', 'ncc', pulse_width=1, n_bits=8,
        #                     account_irf=True, h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 30, circ_shifted=True)),
        # ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1, account_irf=False,
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 1, circ_shifted=True)),
        # ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=8),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', pulse_width=1, account_irf=True,
                            model=os.path.join('bandlimited_models', f'n1024_k8_sigma10'),
                            h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 10, circ_shifted=True)),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', pulse_width=1, account_irf=True,
        #                     model=os.path.join('bandlimited_models', f'n1024_k8_sigma20'),
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 20, circ_shifted=True)),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', pulse_width=1, account_irf=True,
        #                     model=os.path.join('bandlimited_models', f'n1024_k8_sigma30'),
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 30, circ_shifted=True)),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
                            model=os.path.join('bandlimited_peak_models', f'n1024_k8_sigma1_peak030_counts1000'),
                            h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 1, circ_shifted=True)),

    ]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 0.5
    depths = np.arange(0, params['dMax'], dSample)
    total_cycles = params['rep_freq'] * params['T']

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

    coding_matrices = []
    for i in range(len(imaging_schemes)):
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_matrices.append(coding_obj.decode_corrfs)



    max_coefficients = 46

    labels = [1, 5, 10]

    plot_coeff = np.arange(1, max_coefficients + 1, 20)
    fig, axs = plt.subplots(plot_coeff.shape[0]+1, len(coding_matrices), figsize=(30, 16))

    for i, coding_matrix in enumerate(coding_matrices):
        axs[0, i].plot(coding_matrix)
        axs[0, i].set_ylabel('Coding Matrix')
        #axs[0, i].set_title('W')
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])

        counter = 1
        for num_coefficients in range(1, max_coefficients + 1):
            recon = reconstruct_and_get_code_global_top_n(coding_matrix, num_coefficients)
            if num_coefficients in plot_coeff and counter <= plot_coeff.shape[-1]:
                axs[counter, i].plot(recon)
                mse = np.mean((zero_norm_t(coding_matrix) - zero_norm_t(recon)) ** 2)
                #
                axs[counter, i].set_title(f'MSE: {mse}')

                axs[counter, i].set_ylabel(f'{num_coefficients} coef.')
                axs[counter, i].set_xticks([])
                axs[counter, i].set_yticks([])
                counter += 1

    fig.subplots_adjust(wspace=0.05, hspace=0.35)
    fig.savefig(f'Z:\\Research_Users\\David\\Learned Coding Functions Paper\\bandlimited-peak-fourier2.png',
                bbox_inches='tight', dpi=300)

   # plt.show()