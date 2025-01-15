# Python imports
# Library imports
import time

from IPython.core import debugger
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from plot_figures.plot_utils import *

#matplotlib.use('QTkAgg')
breakpoint = debugger.set_trace

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    params = {}
    params['n_tbins'] = 1024
    #params['dMax'] = 5
    #params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 5 * 1e6
    params['rep_tau'] = 1. / params['rep_freq']
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['T'] = 0.1 #intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM



    # params['imaging_schemes'] = [
    #     ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc'),
    #     ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=32),
    #     ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=3, pulse_width=1),
    #     ImagingSystemParams('Greys', 'Gaussian', 'zncc', n_bits=4, pulse_width=1),
    #     #ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1),
    #     #ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=sigma)
    # ]
    filename = 'coded_model.npy'
    params['imaging_schemes'] = [
        ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc', freq_window=0.10, duty=1./12.),
        #ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=32),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=4, pulse_width=1, account_irf=True,
                            freq_window=0.10),
        ImagingSystemParams('Learned', 'Learned', 'zncc', checkpoint_file=filename, pulse_width=1, freq_window=0.10),
        ImagingSystemParams('Greys', 'Gaussian', 'zncc', n_bits=4, pulse_width=1, account_irf=True, freq_window=0.10),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, freq_window=0.10),

    ]

    # params['imaging_schemes'] = [
    #     ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
    #                         duty=1./12., freq_window=0.10, binomial=True, gated=True,
    #                         total_laser_cycles=100_000),
    #     ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
    #                         duty=1. / 30., freq_window=0.10, binomial=True, gated=True,
    #                         total_laser_cycles=100_000),
    #     ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1,
    #                         binomial=True, gated=True, total_laser_cycles=100_000),
    #     ImagingSystemParams('Gated', 'Gaussian', 'zncc', pulse_width=sigma, n_gates=32,
    #                         binomial=True, gated=True, total_laser_cycles=100_000)
    # ]



    params['meanBeta'] = 1e-4
    params['trials'] = 1000
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 1.05
    depths = np.arange(dSample, params['dMax']-dSample, dSample)
    # depths = np.array([105.0])

    #Do either average photon count
    photon_count =  (10 ** 4)
    sbr = 1.0
    #Or peak photon count
    peak_photon_count = None #100
    ambient_count = 10

    total_cycles = params['rep_freq'] * params['T']

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

    if peak_photon_count is not None:
        print(f'Peak Photon count : {peak_photon_count}')
    init_coding_list(n_tbins, depths, params, t_domain=t_domain)
    imaging_schemes = params['imaging_schemes']

    for i in range(len(imaging_schemes)):
        tic = time.perf_counter()
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        if peak_photon_count is not None:
            incident = light_obj.simulate_peak_photons(peak_photon_count, ambient_count)
        else:
            incident = light_obj.simulate_average_photons(photon_count, sbr)

        coded_vals = coding_obj.encode(incident, trials).squeeze()


        if coding_scheme in ['Identity']:
            assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
            decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma,
                                                               rec_algo_id=rec_algo) * tbin_depth_res
        else:
            decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

        errors = np.abs(decoded_depths - depths[np.newaxis, :]) * depth_res
        all_error = np.reshape((decoded_depths - depths[np.newaxis, :]) * depth_res, (errors.size))
        error_metrix = calc_error_metrics(errors)
        print()
        print_error_metrics(error_metrix, prefix=coding_scheme, K=coding_obj.n_functions)

        toc = time.perf_counter()
        print(f'{toc-tic:.6f} seconds')


print()
print('YAYYY')
