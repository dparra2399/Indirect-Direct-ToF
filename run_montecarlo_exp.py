# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace
mpl.use('qt5agg')
from simulate_tof_scene import combined_and_indirect_mae, direct_mae
from research_utils import shared_constants
from direct_toflib import direct_tof_utils, tirf
from direct_toflib.coding_utils import init_coding_list
from indirect_toflib.CodingFunctions_utils import init_coding_functions_list
from research_utils import np_utils
import simulate_tof_scene


def run_experiment(params, depths, sbr_levels, pAveAmbient_levels, pAveSource_levels):

    (n_noise_lvls,n_photon_lvls) = pAveSource_levels.shape
    coding_schemes = params['coding_schemes']
    coding_functions = params['coding_functions']
    pw_factors = params['pw_factors']
    n_tbins = params['n_tbins']
    n_coding_schemes = len(params['coding_schemes'])
    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (direct_tof_utils.calc_tof_domain_params(n_tbins, rep_tau=params['rep_tau']))
    gt_tshifts = direct_tof_utils.depth2time(depths)

    if (len(pw_factors) == 1): pw_factors = np_utils.to_nparray([pw_factors[0]] * n_coding_schemes)

    results = {}

    for k in range(len(coding_functions)):
        #results[coding_functions[k] + '_IDTOF'] = np.zeros(pAveSource_levels.shape)
        results[coding_functions[k] + '_ITOF'] = np.zeros(pAveSource_levels.shape)

    for k in range(len(coding_schemes)):
        results[coding_schemes[k] + '_PP'] = np.zeros(pAveSource_levels.shape)
        results[coding_schemes[k] + '_AVE'] = np.zeros(pAveSource_levels.shape)

    sigma = pw_factors * tbin_res
    pulses_list_pp = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=gt_tshifts, t_domain=t_domain)
    pulses_list_ave = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=gt_tshifts, t_domain=t_domain)

    direct_coding_list = init_coding_list(coding_schemes, n_tbins, params)
    indirect_coding_list = init_coding_functions_list(coding_functions, n_tbins, params)

    assert sbr_levels is None or pAveAmbient_levels is None, "sbr or ambient lists must be none"
    for x in range(0, n_noise_lvls):
        for y in range(0, n_photon_lvls):
            if sbr_levels is not None:
                sbr = sbr_levels[x, y]
            else:
                sbr = None

            if pAveAmbient_levels is not None:
                pAveAmbient = pAveAmbient_levels[x, y]
            else:
                pAveAmbient = None

            pAveSource = pAveSource_levels[x, y]

            (results_indirect, results_direct) = run_both(params, depths, sbr, pAveAmbient, pAveSource,
                                                          indirect_coding_list=indirect_coding_list,
                                                          direct_coding_list=direct_coding_list,
                                                          pulses_list_pp=pulses_list_pp,
                                                          pulses_list_ave=pulses_list_ave)

            for k in range(len(coding_functions)):
                #results[coding_functions[k] + '_IDTOF'][x, y] = results_indirect[coding_functions[k] + '_IDTOF']
                results[coding_functions[k] + '_ITOF'][x, y] = results_indirect[coding_functions[k] + '_ITOF']

            for k in range(len(coding_schemes)):
                results[coding_schemes[k] + '_PP'][x, y] = results_direct[coding_schemes[k] + '_PP']
                results[coding_schemes[k] + '_AVE'][x, y] = results_direct[coding_schemes[k] + '_AVE']


    return results

def run_both(params, depths, sbr, pAveAmbient, pAveSource, indirect_coding_list=None, direct_coding_list=None,
             pulses_list_pp=None, pulses_list_ave=None):

    results_indirect = simulate_tof_scene.combined_and_indirect_mae(params, depths, sbr, pAveAmbient, pAveSource,
                                                                    coding_list=indirect_coding_list)
    results_direct = simulate_tof_scene.direct_mae(params, depths, sbr, pAveAmbient, pAveSource,
                                                   coding_list=direct_coding_list, pulses_list_pp=pulses_list_pp,
                                                   pulses_list_ave=pulses_list_ave)


    return (results_indirect, results_direct)


def gate_size_exp(params, depths, source_photons, sbr):
    min_len = 5.75 * 1e-9
    max_len = 10 * 1e-6 ## typical is 10 * 1e-6

    pw_factors = params['pw_factors']
    n_tbins = params['n_tbins']
    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (direct_tof_utils.calc_tof_domain_params(n_tbins, rep_tau=params['rep_tau']))
    gt_tshifts = direct_tof_utils.depth2time(depths)

    gate_sizes = np.linspace(min_len, 10*1e-9, num=1000)
    gate_sizes = np.arange(1, n_tbins)

    coding_schemes = params['coding_schemes']
    if (len(pw_factors) == 1): pw_factors = np_utils.to_nparray([pw_factors[0]] * len(params['coding_schemes']))

    sigma = pw_factors * tbin_res
    pulses_list_pp = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=gt_tshifts, t_domain=t_domain)
    pulses_list_ave = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=gt_tshifts, t_domain=t_domain)


    error = []
    gates = []
    for i in range(gate_sizes.shape[0]):
        gate_size = gate_sizes[i]
        params['gate_size'] = gate_size * tbin_res
        if (5.75 * 1e-9) <= params['gate_size'] <= (10 * 1e-6):
            direct_coding_list = init_coding_list(coding_schemes, n_tbins, params)
            results_direct = simulate_tof_scene.direct_mae(params, depths, sbr, None, source_photons,
                                                           coding_list=direct_coding_list,
                                                           pulses_list_pp=pulses_list_pp,
                                                           pulses_list_ave=pulses_list_ave)
            error.append(results_direct['IntegratedGated_PP'])
            gates.append(gate_size)

    pulse_width = 'pulse width: ' + str(params['pw_factors'][0])
    plt.scatter(gates, error, label=pulse_width)
    plt.plot(gates[np.argmin(error)], np.min(error), 'ro', color='red')
    print('hello world')

