# Python imports
# Library imports
import time

from IPython.core import debugger
from IPython.core.pylabtools import figsize

from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from plot_figures.plot_utils import *

#matplotlib.use('QTkAgg')
breakpoint = debugger.set_trace
from matplotlib import rc


font = {'family': 'serif',
        'size': 20}

rc('font', **font)

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

    params['meanBeta'] = 1e-4
    params['trials'] = 5000
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 0.5
    depths = np.arange(1.0, params['dMax']-1.0, dSample)
    # depths = np.array([105.0])

    photon_count =  1000
    sbrs = [0.1, 1.0]
    quants = [1, 2, 4, 8, 16, 32, 64]
    #peak_factors = [0.005, 0.015, 0.03]
    peak_factors = [None]
    #sigmas = [1, 5, 10]
    K = 8
    sigmas = [10, 20, 30]


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


    errors_all = np.zeros(((len(peak_factors), len(sigmas), len(quants), 3, 2)))
    for i in range(len(peak_factors)):
        for j in range(len(sigmas)):
            sigma = sigmas[j]
            peak_factor = peak_factors[i]
            for k in range(len(quants)):
                quant = quants[k]

                irf = gaussian_pulse(np.arange(params['n_tbins']), 0, sigma, circ_shifted=True)
                if peak_factors[i] is not None:
                    peak_name = f"{peak_factors[i]:.3f}".split(".")[-1]
                    params['imaging_schemes'] = [
                        ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True,
                                            h_irf=irf),
                        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1,
                                            account_irf=True,
                                            h_irf=irf, quant=quant),
                        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                            model=os.path.join('bandlimited_peak_models',
                                                               f'n1024_k{K}_sigma{sigma}_peak{peak_name}_counts1000'),
                                            account_irf=True, h_irf=irf, quant=quant,),
                    ]
                else:
                    params['imaging_schemes'] = [
                        ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True),
                        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1,
                                            account_irf=True,
                                            h_irf=irf, quant=quant),
                        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                             model=os.path.join('bandlimited_models', f'n1024_k{K}_sigma{sigma}'),
                                            account_irf=True, h_irf=irf, quant=quant,),
                    ]


                init_coding_list(n_tbins, params, t_domain=t_domain)
                imaging_schemes = params['imaging_schemes']

                for l in range(len(imaging_schemes)):
                    imaging_scheme = imaging_schemes[l]
                    coding_obj = imaging_scheme.coding_obj
                    coding_scheme = imaging_scheme.coding_id
                    light_obj = imaging_scheme.light_obj
                    light_source = imaging_scheme.light_id
                    rec_algo = imaging_scheme.rec_algo


                    incident_low, tmp_irf = light_obj.simulate_average_photons(photon_count, sbrs[0], depths, peak_factor=peak_factor)

                    incident_high, _ = light_obj.simulate_average_photons(photon_count, sbrs[1], depths, peak_factor=peak_factor)

                    coded_vals_low = coding_obj.encode(incident_low, trials).squeeze()
                    coded_vals_high = coding_obj.encode(incident_high, trials).squeeze()

                    coding_obj.update_tmp_irf(tmp_irf)
                    coding_obj.update_C_derived_params()

                    decoded_depths_low = coding_obj.max_peak_decoding(coded_vals_low, rec_algo_id=rec_algo) * tbin_depth_res
                    decoded_depths_high = coding_obj.max_peak_decoding(coded_vals_high, rec_algo_id=rec_algo) * tbin_depth_res

                    errors_low = np.abs(decoded_depths_low - depths[np.newaxis, :]) * depth_res
                    error_metrix_low = calc_error_metrics(errors_low)

                    errors_high = np.abs(decoded_depths_high - depths[np.newaxis, :]) * depth_res
                    error_metrix_high = calc_error_metrics(errors_high)

                    errors_all[i, j, k, l, 0] = error_metrix_low['rmse']
                    errors_all[i, j, k, l, 1] = error_metrix_high['rmse']

                    #print(errors_all[i, j, k, l, 0])
                    #print(errors_all[i, j, k, l, 1])


            filename = f'bit_depth_k{K}_bandlimited.npz'
            outfile = './data/results/bit_depth/' + filename

            np.savez(outfile, params=params, results=errors_all, quants=quants, peak_factors=peak_factors, sigmas=sigmas)
print()
print('YAYYY')
