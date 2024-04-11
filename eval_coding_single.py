# Python imports
# Library imports
import time

from IPython.core import debugger
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from spad_toflib import spad_tof_utils
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
import numpy as np

breakpoint = debugger.set_trace

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

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
                                 ImagingSystemParams('IdentitySWISSPAD', 'GaussianSWISSPAD', 'matchfilt', pulse_width=1,
                                                     total_laser_cycles=10000)]
    params['meanBeta'] = 1e-4
    params['trials'] = 1000
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 3.0
    depths = np.arange(3.0, params['dMax'], dSample)
    # depths = np.array([105.0])

    p_ave_source = (10 ** 5.5)
    # pAveAmbient = (10**5)
    p_ave_ambient = None
    sbr = 1

    n_tbins = params['n_tbins']
    mean_beta = params['meanBeta']
    tau = params['rep_tau']
    depth_res = params['depth_res']
    t = params['T']
    trials = params['trials']
    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))

    print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')
    print()

    init_coding_list(n_tbins, depths, params, t_domain=t_domain, pulses_list=None)
    imaging_schemes = params['imaging_schemes']

    tic = time.perf_counter()
    for i in range(len(imaging_schemes)):
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        light_obj.set_all_params(sbr=sbr, ave_source=p_ave_source, ambient=p_ave_ambient, rep_tau=tau,
                                 t=t, mean_beta=mean_beta)

        incident = light_obj.simulate_photons()

        if light_source in ['Gaussian', 'GaussianSWISSPAD']:
            coded_vals = coding_obj.encode_impulse(incident, trials)
        else:
            coded_vals = coding_obj.encode_cw(incident, trials)

        if coding_scheme in ['Identity', 'IdentitySWISSPAD']:
            assert light_source in ['Gaussian', 'GaussianSWISSPAD'], 'Identity coding only available for IRF'
            decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                               rec_algo_id=rec_algo) * tbin_depth_res
        else:
            decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

        imaging_scheme.mean_absolute_error = spad_tof_utils.compute_metrics(depths, decoded_depths) * depth_res

    toc = time.perf_counter()

    for i in range(len(params['imaging_schemes'])):
        scheme = params['imaging_schemes'][i]
        print(f"MAE {scheme.coding_id}: {scheme.mean_absolute_error: .3f} mm,")

print()
print('YAYYY')
