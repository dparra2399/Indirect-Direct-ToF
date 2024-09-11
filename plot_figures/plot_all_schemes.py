import time

from IPython.core import debugger

from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from spad_toflib import spad_tof_utils
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
import numpy as np
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from spad_toflib.emitted_lights import GaussianTIRF
from utils.file_utils import get_string_name
from utils.plot_utils import *
import matplotlib.pyplot as plt


font = {'family': 'serif',
        'size': 7}

matplotlib.rc('font', **font)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    breakpoint = debugger.set_trace

    # Press the green button in the gutter to run the script.
    if __name__ == '__main__':
        params = {}
        params['n_tbins'] = 128
        # params['dMax'] = 5
        # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
        params['rep_freq'] = 5 * 1e6
        params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
        params['rep_tau'] = 1. / params['rep_freq']
        params['T'] = 0.1  # intergration time [used for binomial]
        params['depth_res'] = 1000  ##Conver to MM

        pulse_width = 8e-9
        tbin_res = params['rep_tau'] / params['n_tbins']
        sigma = int(pulse_width / tbin_res)

        params['imaging_schemes'] = [
            ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_freqs=1, pulse_width=sigma),
            ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                                 duty=1. / 4., freq_window=0.10),
            ImagingSystemParams('Identity', 'Gaussian', 'linear', pulse_width=1),
            ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=sigma),
            ImagingSystemParams('Gated', 'Gaussian', 'linear', n_gates=16, pulse_width=sigma)
        ]

        # params['imaging_schemes'] = [
        #     ImagingSystemParams('Greys', 'Gaussian', 'zncc', n_bits=4, pulse_width=sigma),
        #     ImagingSystemParams('HamiltonianK3', 'HamiltonianK3', 'zncc',
        #                          duty=1. / 4., freq_window=0.10),
        #     ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
        #                         duty=1. / 4., freq_window=0.10),
        #     ImagingSystemParams('KTapSinusoid', 'KTapSinusoid', 'zncc',
        #                         ktaps=3),
        #     ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_freqs=2, pulse_width=sigma),
        #     ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_freqs=3, pulse_width=sigma)
        #
        # ]

        params['meanBeta'] = 1e-4
        params['trials'] = 500
        params['freq_idx'] = [1]

        print(f'max depth: {params["dMax"]} meters')
        print()

        dSample = 1.0
        depths = np.arange(dSample, params['dMax'] - dSample, dSample)
        # depths = np.array([105.0])

        total_cycles = params['rep_freq'] * params['T']

        n_tbins = params['n_tbins']
        mean_beta = params['meanBeta']
        tau = params['rep_tau']
        depth_res = params['depth_res']
        t = params['T']
        trials = params['trials']
        (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
            (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))

        init_coding_list(n_tbins, depths, params, t_domain=t_domain)
        imaging_schemes = params['imaging_schemes']

        fig, axs = plt.subplots(nrows=len(imaging_schemes), ncols=4)

        axs[0][1].set_title('Incident Waveform')
        axs[0][2].set_title('Demodulation Functions')
        axs[0][3].set_title('Coding Matrix')
        for i in range(len(imaging_schemes)):
            imaging_scheme = imaging_schemes[i]
            coding_obj = imaging_scheme.coding_obj
            coding_scheme = imaging_scheme.coding_id
            light_obj = imaging_scheme.light_obj
            light_source = imaging_scheme.light_id
            rec_algo = imaging_scheme.rec_algo

            axs[i][1].set_xticks([])
            axs[i][1].set_yticks([])
            axs[i][2].set_xticks([])
            axs[i][2].set_yticks([])
            axs[i][3].set_xticks([])
            axs[i][3].set_yticks([])
            axs[i][1].spines['top'].set_visible(False)
            axs[i][1].spines['right'].set_visible(False)
            axs[i][2].spines['top'].set_visible(False)
            axs[i][2].spines['right'].set_visible(False)
            axs[i][3].spines['top'].set_visible(False)
            axs[i][3].spines['right'].set_visible(False)

            axs[i][1].set_ylabel('Intensity')
            axs[i][1].set_xlabel('Time')
            axs[i][2].set_xlabel('Time')

            if coding_scheme in ['TruncatedFourier', 'Identity', 'Gated', 'Greys']:
                axs[i][2].plot(coding_obj.correlations)
                axs[i][3].imshow(coding_obj.correlations.transpose(), aspect='auto', cmap=plt.cm.get_cmap('binary').reversed())

            else:
                axs[i][2].plot(coding_obj.demodfs)
                axs[i][3].imshow(coding_obj.demodfs.transpose(), aspect='auto', cmap=plt.cm.get_cmap('binary').reversed())


            axs[i][1].plot(np.roll(light_obj.light_source, int(n_tbins/2)), color='blue')
            axs[i][0].set_axis_off()
            axs[i][0].text(0.0, 0.5, f'{get_string_name(imaging_scheme)}')
    #fig.text(0.04, 0.25, 'Hamiltonian', va='center', rotation='vertical', fontsize=7)

    fig.suptitle('Coding Scheme Tested in Paper', fontsize=12,  fontweight="bold")
    fig.tight_layout()  # Adjust the rect to make space for the common labels


    for i in range(len(axs) - 1):
        row_bottom = axs[i, 0].get_position().y0

        # Add horizontal line
        fig.add_artist(plt.Line2D(
            [0, 1], [row_bottom-0.035, row_bottom-0.035], transform=fig.transFigure,
            color='black', linewidth=1
        ))

    row_top = axs[0, 0].get_position().y1

    # Add horizontal line
    fig.add_artist(plt.Line2D(
        [0, 1], [row_top+0.05, row_top+0.05], transform=fig.transFigure,
        color='black', linewidth=1
    ))

    for j in range(len(axs[0])):
        col_left = axs[0, j].get_position().x1
        if j > 0:
            col_left += 0.02
        else:
            col_left -= 0.02
        fig.add_artist(plt.Line2D(
            [col_left, col_left], [0, 0.935], transform=fig.transFigure,
            color='black', linewidth=1
        ))

    fig.savefig('Z:\\Research_Users\\David\\paper figures\\supfigure1.svg', bbox_inches='tight')
    plt.show()

