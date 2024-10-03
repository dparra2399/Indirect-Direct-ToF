# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
import os
import seaborn as sns
sns.set_theme()

breakpoint = debugger.set_trace
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics
from utils.coding_schemes_utils import init_coding_list
from spad_toflib import spad_tof_utils
from utils.coding_schemes_utils import ImagingSystemParams
from utils.file_utils import get_string_name
#mpl.use('TkAgg')



if __name__ == "__main__":

    params = {}
    params['n_tbins'] = 1024
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

    # params['imaging_schemes'] = [
    #     ImagingSystemParams('HamiltonianK3', 'HamiltonianK3', 'zncc',
    #                         duty=1. / 4., freq_window=0.10, binomial=True, gated=True,
    #                         total_laser_cycles=1_000_000),
    #     ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
    #                         duty=1./4., freq_window=0.10, binomial=True, gated=True,
    #                         total_laser_cycles=1_000_000),
    #     ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1,
    #                         binomial=True, gated=True, total_laser_cycles=1_000_000),
    #     ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=sigma,
    #                         binomial=True, gated=True, total_laser_cycles=1_000_000)
    # ]


    params['imaging_schemes'] = [
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_freqs=2, pulse_width=sigma),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_freqs=3, pulse_width=sigma),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_freqs=4, pulse_width=sigma),

        ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                            duty=1. / 6., freq_window=0.10),
    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 100
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()


    n_peak_lvls = 10
    (min_peak_count, max_peak_count) = (5, 30)
    ambient_counts = [25, 5]


    dSample = 1.0
    depths = np.arange(dSample, params['dMax'], dSample)

    n_signals_list = np.round(np.linspace(min_peak_count, max_peak_count, n_peak_lvls))


    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
    gt_tshifts = tof_utils_felipe.depth2time(depths)

    init_coding_list(params['n_tbins'], depths, params, t_domain=t_domain)

    print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')
    print()

    imaging_schemes = params['imaging_schemes']
    trials = params['trials']
    depth_res = params['depth_res']
    results = np.zeros((len(imaging_schemes), n_peak_lvls, 2))

    for pp in range(len(ambient_counts)):
        ambient_count = ambient_counts[pp]
        for i in range(len(imaging_schemes)):
            imaging_scheme = imaging_schemes[i]
            coding_obj = imaging_scheme.coding_obj
            coding_scheme = imaging_scheme.coding_id
            light_obj = imaging_scheme.light_obj
            light_source = imaging_scheme.light_id
            rec_algo = imaging_scheme.rec_algo

            for y in range(0, n_peak_lvls):
                peak_photon_count = n_signals_list[y]
                incident = light_obj.simulate_peak_photons(peak_photon_count, ambient_count)

                coded_vals = coding_obj.encode(incident, trials).squeeze()

                if coding_scheme in ['Identity']:
                    assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
                    decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                                       rec_algo_id=rec_algo) * tbin_depth_res
                else:
                    decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

                errors = np.abs(decoded_depths - depths[np.newaxis, :]) * depth_res
                error_metrix = calc_error_metrics(errors)
                results[i, y, pp] = error_metrix['mae']

    fig, ax = plt.subplots(1, len(ambient_counts), figsize=(15, 5))

    for k in range(len(ambient_counts)):
        for j in range(len(imaging_schemes)):
            ax[k].plot(n_signals_list, results[j, :, k])

            ax[k].scatter(x=n_signals_list, y=results[j, :, k], label=get_string_name(imaging_schemes[j]))

        ax[k].set_xlabel('Peak Photon Count')
        ax[k].set_ylabel('MAE (mm)')
        ax[k].set_ylim(0, 1000)
        ax[k].grid()
        ax[k].legend(loc='upper right')
        ax[k].grid()


    ax[0].set_title(f'{ambient_counts[0]} Ambient Photons Per Bin')
    ax[1].set_title(f'{ambient_counts[1]} Ambient Photons Per Bin')

    save_folder = 'Z:\\Research_Users\\David\\paper figures'
    #fig.savefig(os.path.join(save_folder, 'suppfigure6.svg'), bbox_inches='tight')
    plt.show()
    print('complete')
