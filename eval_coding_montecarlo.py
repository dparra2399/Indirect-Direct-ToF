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
from utils.file_utils import write_errors_to_file

if __name__ == "__main__":

    params = {}
    params['n_tbins'] = 1000
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 1 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['gate_size'] = 1 * ((1. / params['rep_freq']) / params['n_tbins'])
    params['T'] = 0.1  # Integration time. Exposure time in seconds
    params['rep_tau'] = 1. / params['rep_freq']
    params['depth_res'] = 1000  ##Conver to MM

    params['imaging_schemes'] = [ImagingSystemParams('HamiltonianK3', 'HamiltonianK3', 'zncc',
                                                     binomial=True, total_laser_cycles=6_000_000),
                                 ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
                                                     binomial=True, total_laser_cycles=6_000_000),
                                 ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                                                     binomial=True, total_laser_cycles=6_000_000),
                                 ImagingSystemParams('Identity', 'Gaussian', 'linear', pulse_width=1,
                                                     binomial=True, total_laser_cycles=6_000_000),
                                 ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=30,
                                                     n_gates=250, binomial=True, total_laser_cycles=6_000_000)
                                 ]
    params['meanBeta'] = 1e-4
    params['trials'] = 1000
    params['freq_idx'] = [1]

    params['levels_one'] = 'laser cycles'
    params['levels_one_exp'] = (3, 7)
    params['num_levels_one'] = 10
    params['levels_two'] = 'ave power'
    params['levels_two_exp'] = (2, 6)
    params['num_levels_two'] = 10

    n_level_one = params['num_levels_one']
    n_level_two = params['num_levels_two']

    ave_source = 10 ** 5
    sbr = 1
    laser_cycles = 5000

    dSample = 3.0
    depths = np.arange(3.0, params['dMax'], dSample)

    (levels_one, levels_two) = get_levels_list_montecarlo(params)

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
    gt_tshifts = tof_utils_felipe.depth2time(depths)

    init_coding_list(params['n_tbins'], depths, params, t_domain=t_domain, pulses_list=None)

    imaging_schemes = params['imaging_schemes']
    trials = params['trials']
    t = params['T']
    mean_beta = params['meanBeta']
    results = np.zeros((len(imaging_schemes), n_level_one, n_level_two))

    updated_params = {'laser cycles': laser_cycles,
                      'integration time': t,
                      'ave power': ave_source,
                      'sbr': sbr}
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

                light_obj.set_all_params(sbr=updated_params['sbr'], ave_source=updated_params['ave power'],
                                         rep_tau=rep_tau, t=updated_params['integration time'], mean_beta=mean_beta)

                incident = light_obj.simulate_photons()

                coding_obj.set_laser_cycles(updated_params['laser cycles'])
                if light_source in ['Gaussian']:
                    coded_vals = coding_obj.encode_impulse(incident, trials)
                else:
                    coded_vals = coding_obj.encode_cw(incident, trials)

                if coding_scheme in ['Identity']:
                    assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
                    decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                                       rec_algo_id=rec_algo) * tbin_depth_res
                else:
                    decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

                imaging_scheme.mean_absolute_error = spad_tof_utils.compute_metrics(depths, decoded_depths) * params[
                    'depth_res']
                results[i, y, x] = imaging_scheme.mean_absolute_error

    write_errors_to_file(params, results, depths, levels_one=levels_one, levels_two=levels_two)
    print('complete')
