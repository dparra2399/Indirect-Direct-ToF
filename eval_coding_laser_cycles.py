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
from utils.coding_schemes_utils import ImagingSystemParams

if __name__ == "__main__":

    params = {}
    params['n_tbins'] = 300
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 1 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['gate_size'] = 1 * ((1. / params['rep_freq']) / params['n_tbins'])
    params['T'] = 0.1  # Integration time. Exposure time in seconds
    params['rep_tau'] = 1. / params['rep_freq']
    params['depth_res'] = 1000  ##Conver to MM

    params['imaging_schemes'] = [ImagingSystemParams('HamiltonianK3SWISSPAD', 'HamiltonianK3SWISSPAD', 'zncc',
                                                     total_laser_cycles=10000),
                                 ImagingSystemParams('HamiltonianK4SWISSPAD', 'HamiltonianK3SWISSPAD', 'zncc',
                                                     total_laser_cycles=10000),
                                 ImagingSystemParams('HamiltonianK5SWISSPAD', 'HamiltonianK3SWISSPAD', 'zncc',
                                                     total_laser_cycles=10000),
                                 ImagingSystemParams('IdentitySWISSPAD', 'GaussianSWISSPAD', 'matchfilt', pulse_width=1,
                                                     total_laser_cycles=10000)]
    params['meanBeta'] = 1e-4
    params['trials'] = 1000
    params['freq_idx'] = [1]

    # n_signal_lvls = 20
    n_cycles = 10
    # n_sbr = 1

    p_ave_source = 10 ** 6
    sbr = 1
    p_ave_ambient = None
    (min_cycles_exp, max_cycles_exp) = (3, 6)
    # (min_sbr_exp, max_sbr_exp) = (1, 1)

    dSample = 3.0
    depths = np.arange(3.0, params['dMax'], dSample)

    n_cycles_list = np.round(np.power(10, np.linspace(min_cycles_exp, max_cycles_exp, n_cycles)))
    # sbr_list = np.power(10, np.linspace(min_sbr_exp, max_sbr_exp, n_sbr))
    # sbr_levels, n_cycles_levels = np.meshgrid(sbr_list, n_cycles_list)

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
    gt_tshifts = tof_utils_felipe.depth2time(depths)

    init_coding_list(params['n_tbins'], depths, params, t_domain=t_domain, pulses_list=None)

    imaging_schemes = params['imaging_schemes']
    trials = params['trials']
    results = np.zeros((len(imaging_schemes), n_cycles))

    for i in range(len(imaging_schemes)):
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        for y in range(0, n_cycles):
            laser_cycles = n_cycles_list[y]
            light_obj.set_all_params(sbr=sbr, ave_source=p_ave_source, ambient=p_ave_ambient,
                                     rep_tau=params['rep_tau'],
                                     t=params['T'], mean_beta=params['meanBeta'])

            incident = light_obj.simulate_photons()

            coding_obj.set_laser_cycles(laser_cycles)
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
            results[i, y] = imaging_scheme.mean_absolute_error

    for j in range(len(imaging_schemes)):
        plt.plot(np.log10(n_cycles_list), results[j, :])
        plt.scatter(x=np.log10(n_cycles_list), y=results[j, :], label=imaging_schemes[j].coding_id)

    plt.legend()
    plt.xlabel('Number of total laser cycles (log)')
    plt.ylabel('MAE (mm)')
    plt.grid()
    plt.show()
    print('complete')
