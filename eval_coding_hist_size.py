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
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 1 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    # params['gate_size'] = 1 * ((1. / params['rep_freq']) / params['n_tbins'])
    params['T'] = 0.1  # Integration time. Exposure time in seconds
    params['rep_tau'] = 1. / params['rep_freq']
    params['depth_res'] = 1000  ##Conver to MM

    params['imaging_schemes'] = [ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                                                     binomial=True, total_laser_cycles=6_000_000),
                                 ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
                                                     binomial=True, total_laser_cycles=6_000_000),
                                 ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1,
                                                     binomial=True, total_laser_cycles=6_000_000)]

    params['meanBeta'] = 1e-4
    params['trials'] = 1000
    params['freq_idx'] = [1]

    p_ave_source = 10 ** 5
    sbr = 1
    p_ave_ambient = None

    # Get Depths and Shift to Time Scale
    dSample = 3.0
    depths = np.arange(3.0, params['dMax'], dSample)
    gt_tshifts = tof_utils_felipe.depth2time(depths)

    (min_time_bins, max_time_bins) = (1000, 9_000)
    time_bin_sample = 1000
    n_tbins_list = np.arange(min_time_bins, max_time_bins, time_bin_sample)

    trials = params['trials']
    results = np.zeros((len(params['imaging_schemes']), n_tbins_list.shape[0]))

    for y in range(0, n_tbins_list.shape[0]):
        n_tbins = n_tbins_list[y]
        params['n_tbins'] = n_tbins
        (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
            (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
        init_coding_list(n_tbins, depths, params, t_domain=t_domain, pulses_list=None)
        imaging_schemes = params['imaging_schemes']
        for i in range(len(imaging_schemes)):
            imaging_scheme = imaging_schemes[i]
            coding_obj = imaging_scheme.coding_obj
            coding_scheme = imaging_scheme.coding_id
            light_obj = imaging_scheme.light_obj
            light_source = imaging_scheme.light_id
            rec_algo = imaging_scheme.rec_algo
            light_obj.set_all_params(sbr=sbr, ave_source=p_ave_source, ambient=p_ave_ambient,
                                     rep_tau=params['rep_tau'],
                                     t=params['T'], mean_beta=params['meanBeta'])

            incident = light_obj.simulate_photons()

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
        plt.plot(n_tbins_list, results[j, :])
        plt.scatter(x=n_tbins_list, y=results[j, :], label=imaging_schemes[j].coding_id)

    plt.legend()
    plt.xlabel('Number of Time Bins')
    plt.ylabel('MAE (mm)')
    plt.grid()
    plt.show()
    print('complete')
