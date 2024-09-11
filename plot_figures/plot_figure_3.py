import time

from IPython.core import debugger
from IPython.core.pylabtools import figsize

from felipe_utils.felipe_impulse_utils.tof_utils_felipe import depth2time
from spad_toflib.spad_tof_utils import normalize_measure_vals
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
        params['n_tbins'] = 2048
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
            ImagingSystemParams('HamiltonianK3', 'HamiltonianK3', 'zncc',
                                duty=1. / 6., freq_window=0.1)
        ]

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

        imaging_scheme = imaging_schemes[0]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        incident = light_obj.simulate_peak_photons(10, 2)

        coded_vals = coding_obj.encode(incident, trials).squeeze()

        decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res


        inc = incident[18, 0, :]
        b_vals = normalize_measure_vals(coded_vals[100, 18, :])
        d_hat = int(decoded_depths[100, 18] / tbin_depth_res)
        correlations = coding_obj.correlations
        demodfs = coding_obj.demodfs

        fig, axs = plt.subplots(1, 3, figsize=(6, 2))

        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].set_ylim(0, 15)
        axs[0].set_title('Modulation Function')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Intensity')

        axs[0].plot(inc, color='blue')

        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].plot(demodfs[:, 0], color='salmon')
        axs[1].plot(demodfs[:, 1], color='limegreen')
        axs[1].plot(demodfs[:, 2], color='violet')
        axs[1].set_title('Demodulation Functions')
        axs[1].set_xlabel('Time')

        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].set_title('Correlation Functions')
        axs[2].set_xlabel('Depth')
        axs[2].plot(correlations[:, 0], color='salmon')
        axs[2].plot(correlations[:, 1], color='limegreen')
        axs[2].plot(correlations[:, 2], color='violet')

        fig.tight_layout()
        fig.savefig(os.path.join(save_folder, 'figure3a.svg'), bbox_inches='tight')

        plt.show()