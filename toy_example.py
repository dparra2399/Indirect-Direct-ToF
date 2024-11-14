# Python imports
# Library imports
import time

import numpy as np
from IPython.core import debugger
from torch.fft import fftfreq

from felipe_utils.CodingFunctionsFelipe import TauDefault
from spad_toflib.spad_tof_utils import poisson_noise_array
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from plot_figures.plot_utils import *
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse, smooth_codes
from spad_toflib.spad_tof_utils import *
from felipe_utils.research_utils import signalproc_ops, np_utils
from felipe_utils.CodingFunctionsFelipe import *
from scipy.fft import fft



matplotlib.use('TkAgg')
breakpoint = debugger.set_trace

plot = True
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    N = 2000
    K = 3
    start_idx = 1

    dSample = 1.0
    rep_freq = 10 * 1e6
    rep_tau = 1. / rep_freq

    photon_count = 0.5 * (10 ** 4)
    sbr = 1.0


    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(N, rep_tau=rep_tau))

    depths = np.arange(0, dMax, dSample)

    modFs = np.zeros((N,1))
    t = np.linspace(0, 2 * np.pi, N)
    cosF = (0.5 * np.cos(t)) + 0.5
    modFs[:, 0] = cosF

    fourier_mat = signalproc_ops.get_fourier_mat(n=N, freq_idx=np.arange(start_idx, start_idx+K))
    coding_matrix = np.zeros((N, fourier_mat.shape[1]))

    for i in range(fourier_mat.shape[1]):
        if ((i % 2) == 0):
            coding_matrix[:, i] = fourier_mat[:, i // 2].real
        else:
            coding_matrix[:, i] = fourier_mat[:, i // 2].imag

    (_, coding_matrix) = GetCosCos(N, K)
    #coding_matrix = np.stack((np.cos(t), np.sin(t)), axis=-1)
    gaussian_tirf = gaussian_pulse(t_domain, 0, tbin_res, circ_shifted=True)
    gaussian_tirf = np.expand_dims(gaussian_tirf, axis=-1)

    incident_gaussian = simulate_average_photons_n_cycles(gaussian_tirf, total_photons=photon_count, sbr=sbr)
    incident_gaussian = phase_shifted(incident_gaussian, depths, tbin_depth_res)
    incident_modfs = simulate_average_photons_n_cycles(modFs, total_photons=photon_count, sbr=sbr)
    incident_modfs = phase_shifted(incident_modfs, depths, tbin_depth_res)

    noisy_incident_gaussian = poisson_noise_array(incident_gaussian, trials=5000)
    noisy_incident_modfs = poisson_noise_array(incident_modfs, trials=5000)

    c_vals_gaussian = np.einsum('mnp,pq->mnq', np.squeeze(noisy_incident_gaussian), coding_matrix)
    c_vals_modfs = np.einsum('mnp,pq->mnq', np.squeeze(noisy_incident_modfs), coding_matrix)

    correlations = np.fft.ifft(np.fft.fft(modFs, axis=0).conj() * np.fft.fft(coding_matrix, axis=0), axis=0).real

    decoded_depths_gaussian = np.argmax(zncc_reconstruction(c_vals_gaussian, coding_matrix), axis=-1) * tbin_depth_res
    decoded_depths_modfs = np.argmax(zncc_reconstruction(c_vals_modfs, correlations), axis=-1) * tbin_depth_res

    errors_gaussian = np.mean(np.abs(decoded_depths_gaussian - depths[np.newaxis, :])) * 1000
    errors_modfs = np.mean(np.abs(decoded_depths_modfs - depths[np.newaxis, :])) * 1000

    print(f'Pulsed MAE {errors_gaussian:.2f}mm')
    print(f'Sinusoid MAE {errors_modfs:.2f}mm')



    if plot:
        fig, axs = plt.subplots(1, 6, figsize=(20, 4))
        noisy_sig_modfs = noisy_incident_modfs[100, 10, 0, :]
        clean_sig_modfs = incident_modfs[10, ...,]
        axs[0].plot(np.squeeze(clean_sig_modfs))
        axs[0].bar(np.arange(0, N), noisy_sig_modfs, alpha=0.6,  edgecolor='black', linewidth=0.5, color='blue')
        axs[0].set_ylim(0, 20)
        axs[0].set_title('Incident Sinusoid')
        axs[0].set_xlabel('Time Bins')
        axs[0].set_ylabel('Photon Count')

        noisy_sig_gaussian = noisy_incident_gaussian[100, 10, 0, :]
        clean_sig_gaussian = incident_gaussian[10, ...,]
        axs[1].plot(np.squeeze(clean_sig_gaussian))
        axs[1].bar(np.arange(0, N), noisy_sig_gaussian, alpha=0.6,  edgecolor='black', linewidth=0.5, color='blue')
        axs[1].set_ylim(0, 20)
        axs[1].set_title('Incident Pulsed')
        axs[1].set_xlabel('Time Bins')
        axs[1].set_ylabel('Photon Count')

        axs[2].plot(coding_matrix)
        axs[2].set_title('Coding Matrix')
        axs[2].set_xlabel('Time Bins')

        axs[3].bar(0, errors_gaussian, color='green', label=f'{errors_gaussian:.2f}mm')
        axs[3].bar(1, errors_modfs, color='red', label=f'{errors_modfs:.2f}mm')
        axs[3].set_title('MAE (Lower Better)')
        axs[3].set_ylabel('MAE (mm)')
        axs[3].set_xticks([0, 1])
        axs[3].set_xticklabels(['Pulsed', 'Sinusoid'])
        axs[3].legend()

        fft_modfs = np.squeeze(np.fft.fft(noisy_sig_modfs))
        fft_modfs[0] = 0

        freq_spec_modfs = abs(fft_modfs) * (2/N)

        fft_gaussian = np.squeeze(np.fft.fft(noisy_sig_gaussian))
        fft_gaussian[0] = 0
        freq_spec_gaussian = abs(fft_gaussian) * (2/N)

        freq = np.fft.fftfreq(N, 1/rep_freq)


        axs[4].plot(freq[:N//2], freq_spec_gaussian[:N//2], color='red', label='Gaussian')
        axs[4].plot(freq[:N // 2], freq_spec_modfs[:N // 2], color='green', label='Sinusoid')
        axs[4].set_title('Magnitude Spectrum')
        axs[4].set_ylabel('Amplitude')
        axs[4].set_xlabel('Freq')
        axs[4].legend()

        # axs[4].magnitude_spectrum(np.squeeze(clean_sig_gaussian), color='green', label='Pulse') #, label=f'{amp_gaussian:.2f}')
        # axs[4].magnitude_spectrum(np.squeeze(clean_sig_modfs), color='red', label='Sinusoid') #, label=f'{amp_modfs:.2f}')
        # axs[4].set_title('Magnitude Spectrum')
        # axs[4].set_ylabel('Amplitude')
        # axs[4].set_xlabel('Freq')
        # axs[4].legend()

        plt.legend()
        plt.show(block=True)


