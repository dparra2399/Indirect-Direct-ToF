import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
import os
from joblib import Parallel, delayed
breakpoint = debugger.set_trace
from felipe_utils import tof_utils_felipe
from utils.coding_schemes_utils import init_coding_list
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from spad_toflib import spad_tof_utils
from utils.coding_schemes_utils import ImagingSystemParams, get_levels_list_montecarlo
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from utils.file_utils import write_errors_to_file

params = {}
params['n_tbins'] = 1024
# params['dMax'] = 5
# params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
params['rep_freq'] = 5 * 1e6
params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
params['gate_size'] = 1 * ((1. / params['rep_freq']) / params['n_tbins'])
params['T'] = 0.1  # Integration time. Exposure time in seconds
params['rep_tau'] = 1. / params['rep_freq']
params['depth_res'] = 1000  ##Conver to MM

pulse_width = .8e-8
tbin_res = params['rep_tau'] / params['n_tbins']
sigma = int(pulse_width / tbin_res)

sigma = 1
peak_factor = None
metric = 'mae'
k = 4
if peak_factor is not None:
    peak_name =  f"{peak_factor:.3f}".split(".")[-1]
quant = None
gated = True
binomial = False

irf = gaussian_pulse(np.arange(params['n_tbins']), 0, sigma, circ_shifted=True)
constant_pulse_energy = True

print(f'sigma: {sigma}')
print(f'metric: {metric}')
print(f'peak factor: {peak_factor}')

params['imaging_schemes'] = [
    ImagingSystemParams('Gated', 'Gaussian', 'zncc', pulse_width=50, n_gates=8, h_irf=irf, account_irf=True,
                        gated=gated, binomial=binomial),
    # ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=4, pulse_width=1, account_irf=True, h_irf=irf,
    #                     gated=gated, binomial=False),

    # ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', pulse_width=1, n_codes=8, h_irf=irf, account_irf=True,
    #                    gated=False, binomial=False),
    ImagingSystemParams('HamiltonianK3', 'HamiltonianK3', 'zncc', duty=1 / 6, h_irf=irf, account_irf=True,
                        gated=gated, binomial=binomial),
    ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc', duty=1 / 12, h_irf=irf, account_irf=True,
                        gated=gated, binomial=binomial),
    ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc', duty=1 / 30, h_irf=irf,
                        account_irf=True,
                        gated=gated, binomial=binomial),
    # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
    #                    model=os.path.join('bandlimited_binary_models', f'version_1'),
    #                    gated=gated, account_irf=True, h_irf=irf),

]

params['meanBeta'] = 1e-4
params['trials'] = 1000
params['freq_idx'] = [1]

params['levels_one'] = 'ave power'
params['levels_one_exp'] = (1, 3)
params['num_levels_one'] = 25
params['levels_two'] = 'sbr'
params['levels_two_exp'] = (-1, 1)
params['num_levels_two'] = 25

n_level_one = params['num_levels_one']
n_level_two = params['num_levels_two']


dSample = 0.5
depths = np.arange(1.0, params['dMax']-1.0, dSample)

(levels_one, levels_two) = get_levels_list_montecarlo(params)

(rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
    (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
gt_tshifts = tof_utils_felipe.depth2time(depths)

init_coding_list(params['n_tbins'], params, t_domain=t_domain)

imaging_schemes = params['imaging_schemes']
trials = params['trials']
t = params['T']
mean_beta = params['meanBeta']
depth_res = params['depth_res']

updated_params = {'laser cycles': None,
                  'integration time': t,
                  'ave power': None,
                  'sbr': None,
                  'peak power': None,
                  'amb photons': None}

def getHog(imaging_scheme):
    results = np.zeros((n_level_one, n_level_two))
    coding_obj = imaging_scheme.coding_obj
    coding_scheme = imaging_scheme.coding_id
    light_obj = imaging_scheme.light_obj
    light_source = imaging_scheme.light_id
    rec_algo = imaging_scheme.rec_algo
    print(f'running scheme {coding_scheme}')
    for x in range(0, n_level_one):
        for y in range(0, n_level_two):
            updated_params[params['levels_one']] = levels_one[y, x]
            updated_params[params['levels_two']] = levels_two[y, x]

            if imaging_scheme.constant_pulse_energy:
                incident, tmp_irf = light_obj.simulate_constant_pulse_energy(updated_params['ave power'], updated_params['sbr'], depths, peak_factor=peak_factor)
            else:
                incident, tmp_irf = light_obj.simulate_average_photons(updated_params['ave power'], updated_params['sbr'], depths, peak_factor=peak_factor)

            coded_vals = coding_obj.encode(incident, trials, None).squeeze()

            if light_source in 'Gaussian':
                coding_obj.update_tmp_irf(tmp_irf)
                coding_obj.update_C_derived_params()

            if coding_scheme in ['Identity'] and False:
                assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
                decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                                   rec_algo_id=rec_algo) * tbin_depth_res
            else:
                decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

            errors = np.abs(decoded_depths - depths[np.newaxis, :]) * depth_res
            error_metrix = calc_error_metrics(errors)
            results[y, x] = error_metrix[metric]
    return results

output_generator = Parallel(n_jobs=10)(delayed(getHog)(i) for i in imaging_schemes)

results = np.zeros((len(imaging_schemes), n_level_one, n_level_two))
for i, mae in enumerate(output_generator):
    results[i,:, :]=mae
results.shape

# if constant_pulse_energy:
#     exp_num = f'Learned_n{params["n_tbins"]}_k{k}_peak{peak_name}_{metric}_constant_pulse_energy'
# elif peak_factor is not None:
#     if quant:
#         exp_num = f'Learned_n{params["n_tbins"]}_k{k}_sigma{sigma}_peak{peak_name}_{metric}'
#     else:
#         exp_num = f'Learned_n{params["n_tbins"]}_k{k}_sigma{sigma}_peak{peak_name}_{metric}'
# else:
#     if quant:
#         exp_num = f'Learned_n{params["n_tbins"]}_k{k}_sigma{sigma}_{metric}'
#     else:
#         exp_num = f'Learned_n{params["n_tbins"]}_k{k}_sigma{sigma}_{metric}'

exp_num = 'tmp2'
write_errors_to_file(params, results, depths, levels_one=levels_one, levels_two=levels_two, exp=exp_num)
print('complete')

