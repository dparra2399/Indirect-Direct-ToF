# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger

breakpoint = debugger.set_trace
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
from utils.coding_schemes_utils import init_coding_list
from spad_toflib import spad_tof_utils
from utils.coding_schemes_utils import ImagingSystemParams, get_levels_list_montecarlo
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from utils.file_utils import write_errors_to_file
from utils.plot_utils import *

if __name__ == "__main__":

    params = {}
    params['n_tbins'] = 1024
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 10 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['gate_size'] = 1 * ((1. / params['rep_freq']) / params['n_tbins'])
    params['T'] = 0.1  # Integration time. Exposure time in seconds
    params['rep_tau'] = 1. / params['rep_freq']
    params['depth_res'] = 1000  ##Conver to MM

    pulse_width = .8e-8
    tbin_res = params['rep_tau'] / params['n_tbins']
    sigma = int(pulse_width / tbin_res)

    # params['imaging_schemes'] = [
    #     ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_freqs=1, pulse_width=sigma),
    #     ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
    #                         duty=1. / 4., freq_window=0.10),
    #     ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1),
    #     ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=sigma),
    #     ImagingSystemParams('Gated', 'Gaussian', 'linear', n_gates=32, pulse_width=sigma)
    # ]
    params['imaging_schemes'] = [
        ImagingSystemParams('KTapSinusoid', 'KTapSinusoid', 'zncc', ktaps=3, cw_tof=True),
        ImagingSystemParams('KTapSinusoid', 'KTapSinusoid', 'zncc', ktaps=3),

    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 1000
    params['freq_idx'] = [1]

    params['levels_one'] = 'peak power'
    params['levels_one_exp'] = (5, 30)
    params['num_levels_one'] = 20
    params['levels_two'] = 'amb photons'
    params['levels_two_exp'] = (1, 15)
    params['num_levels_two'] = 20

    n_level_one = params['num_levels_one']
    n_level_two = params['num_levels_two']


    dSample = 1.0
    depths = np.arange(dSample, params['dMax']-dSample, dSample)

    (levels_one, levels_two) = get_levels_list_montecarlo(params)

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
    gt_tshifts = tof_utils_felipe.depth2time(depths)

    init_coding_list(params['n_tbins'], depths, params, t_domain=t_domain)

    imaging_schemes = params['imaging_schemes']
    trials = params['trials']
    t = params['T']
    mean_beta = params['meanBeta']
    depth_res = params['depth_res']
    results = np.zeros((len(imaging_schemes), n_level_one, n_level_two))
    probs = np.zeros((n_level_one, n_level_two))
    updated_params = {'laser cycles': None,
                      'integration time': t,
                      'ave power': None,
                      'sbr': None,
                      'peak power': None,
                      'amb photons': None}

    for i in range(len(imaging_schemes)):
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        for x in range(0, n_level_one):
            for y in range(0, n_level_two):
                updated_params[params['levels_one']] = levels_one[y, x]
                updated_params[params['levels_two']] = levels_two[y, x]

                if updated_params['peak power'] is not None:
                    incident = light_obj.simulate_peak_photons(updated_params['peak power'], updated_params['amb photons'])
                else:
                    incident = light_obj.simulate_average_photons(updated_params['ave power'], updated_params['sbr'])

                coded_vals = coding_obj.encode(incident, trials).squeeze()

                if coding_scheme in ['Identity']:
                    assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
                    decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                                       rec_algo_id=rec_algo) * tbin_depth_res
                else:
                    decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

                errors = np.abs(decoded_depths - depths[np.newaxis, :]) * depth_res
                error_metrix = calc_error_metrics(errors)
                results[i, y, x] = error_metrix['mae']
        print('done')


    exp_num = 'sinusoid001'
    write_errors_to_file(params, results, depths, levels_one=levels_one, levels_two=levels_two, exp=exp_num)
    print('complete')
