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
from utils.file_utils import get_string_name

if __name__ == "__main__":

    params = {}
    params['n_tbins'] = 2200
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 1 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['gate_size'] = 1 * ((1. / params['rep_freq']) / params['n_tbins'])
    params['T'] = 0.1  # Integration time. Exposure time in seconds
    params['rep_tau'] = 1. / params['rep_freq']
    params['depth_res'] = 1000  ##Conver to MM


    pulse_width = 8e-9
    tbin_res = params['rep_tau'] / params['n_tbins']
    sigma = int(pulse_width / tbin_res)

    params['imaging_schemes'] = [
        ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
                            duty=30.0, freq_window=0.10, binomial=True, gated=True,
                            total_laser_cycles=100_000_000),
        ImagingSystemParams('HamiltonianK3', 'HamiltonianK3', 'zncc',
                            duty=6.0, freq_window=0.10, binomial=True, gated=True,
                            total_laser_cycles=100_000_000),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=5,
                            binomial=True, gated=True, total_laser_cycles=100_000_000),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=sigma,
                            binomial=True, gated=True, total_laser_cycles=100_000_000)
    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 100
    params['freq_idx'] = [1]

    # n_signal_lvls = 20
    n_cycles = 20
    # n_sbr = 1

    p_ave_source = 10 ** 4
    sbr = 5
    p_ave_ambient = None
    (min_cycles_exp, max_cycles_exp) = (5, 8)
    # (min_sbr_exp, max_sbr_exp) = (1, 1)

    dSample = 1.0
    depths = np.arange(1.0, params['dMax'], dSample)

    n_cycles_list = np.round(np.power(10, np.linspace(min_cycles_exp, max_cycles_exp, n_cycles)))
    # sbr_list = np.power(10, np.linspace(min_sbr_exp, max_sbr_exp, n_sbr))
    # sbr_levels, n_cycles_levels = np.meshgrid(sbr_list, n_cycles_list)

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
    gt_tshifts = tof_utils_felipe.depth2time(depths)

    print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')
    print()

    init_coding_list(params['n_tbins'], depths, params, t_domain=t_domain)

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
        plt.ylim(0, 2000)
        plt.scatter(x=np.log10(n_cycles_list), y=results[j, :], label=get_string_name(imaging_schemes[j]))

    plt.legend()
    plt.xlabel('Number of total laser cycles (log)')
    plt.ylabel('MAE (mm)')
    plt.grid()
    plt.show()
    print('complete')
