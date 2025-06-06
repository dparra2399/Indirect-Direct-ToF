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

    photon_count =  500
    sbrs = [0.1, 1.0]
    #quants = [64, 32, 16, 8, 4, 2, 1]
    quants = [1, 2, 4, 8, 16, 32, 64]
    peak_factor = 0.015

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


    error_arr_bandlimited = np.zeros((len(quants), len(sbrs), 3))
    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 10, circ_shifted=True)

    for j in range(len(quants)):
        quant = quants[j]
        params['imaging_schemes'] = [
            ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf),

            ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1,  account_irf=True,
                                h_irf=irf, quant=quant),
            ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                 model=os.path.join('bandlimited_models', 'n1024_k8_sigma10'),
                                account_irf=True, h_irf=irf, quant=quant),
        ]



        init_coding_list(n_tbins, params, t_domain=t_domain)
        imaging_schemes = params['imaging_schemes']

        for k in range(len(sbrs)):
            for i in range(len(imaging_schemes)):
                imaging_scheme = imaging_schemes[i]
                coding_obj = imaging_scheme.coding_obj
                coding_scheme = imaging_scheme.coding_id
                light_obj = imaging_scheme.light_obj
                light_source = imaging_scheme.light_id
                rec_algo = imaging_scheme.rec_algo

                if imaging_scheme.constant_pulse_energy:
                    incident, tmp_irf = light_obj.simulate_constant_pulse_energy(photon_count, sbrs[k], depths)
                else:
                    incident, tmp_irf = light_obj.simulate_average_photons(photon_count, sbrs[k], depths)

                print(f'\nphoton_count : {np.sum(incident[0, 0, :] - ((photon_count / sbrs[k]) / n_tbins))}')

                coded_vals = coding_obj.encode(incident, trials).squeeze()

                #coding_obj.update_irf(tmp_irf)
                #coding_obj.update_C_derived_params()

                #coded_vals = coding_obj.encode_no_noise(incident).squeeze()

                if coding_scheme in ['wIdentity']:
                    #assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
                    decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma * tbin_depth_res,
                                                                       rec_algo_id=rec_algo) * tbin_depth_res
                else:
                    decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

                errors = np.abs(decoded_depths - depths[np.newaxis, :]) * depth_res
                all_error = np.reshape((decoded_depths - depths[np.newaxis, :]) * depth_res, (errors.size))
                error_metrix = calc_error_metrics(errors)

                error_arr_bandlimited[j, k, i] = error_metrix['rmse']

    error_arr_peaklimited = np.zeros((len(quants), len(sbrs), 3))
    for j in range(len(quants)):
        quant = quants[j]
        params['imaging_schemes'] = [
            ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf),

            ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1,  account_irf=True,
                                h_irf=irf, quant=quant),
            ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
                                model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak015_counts1000'),
                                h_irf=irf, quant=quant),
        ]

        init_coding_list(n_tbins, params, t_domain=t_domain)
        imaging_schemes = params['imaging_schemes']

        for k in range(len(sbrs)):
            for i in range(len(imaging_schemes)):
                imaging_scheme = imaging_schemes[i]
                coding_obj = imaging_scheme.coding_obj
                coding_scheme = imaging_scheme.coding_id
                light_obj = imaging_scheme.light_obj
                light_source = imaging_scheme.light_id
                rec_algo = imaging_scheme.rec_algo

                if imaging_scheme.constant_pulse_energy:
                    incident, tmp_irf = light_obj.simulate_constant_pulse_energy(photon_count, sbrs[k], depths, peak_factor=peak_factor)
                else:
                    incident, tmp_irf = light_obj.simulate_average_photons(photon_count, sbrs[k], depths, peak_factor=peak_factor)

                print(f'\nphoton_count : {np.sum(incident[0, 0, :] - ((photon_count / sbrs[k]) / n_tbins))}')

                coded_vals = coding_obj.encode(incident, trials).squeeze()

                coding_obj.update_irf(tmp_irf)
                coding_obj.update_C_derived_params()

                #coded_vals = coding_obj.encode_no_noise(incident).squeeze()

                if coding_scheme in ['wIdentity']:
                    #assert light_source in ['Gaussian'], 'Identity coding only available for IRF'
                    decoded_depths = coding_obj.maxgauss_peak_decoding(coded_vals, light_obj.sigma * tbin_depth_res,
                                                                       rec_algo_id=rec_algo) * tbin_depth_res
                else:
                    decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

                errors = np.abs(decoded_depths - depths[np.newaxis, :]) * depth_res
                all_error = np.reshape((decoded_depths - depths[np.newaxis, :]) * depth_res, (errors.size))
                error_metrix = calc_error_metrics(errors)

                error_arr_peaklimited[j, k, i] = error_metrix['rmse']

    fig, axs = plt.subplots(2, len(sbrs), squeeze=False, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(len(sbrs)):
        for j in range(3):

            if imaging_schemes[j].coding_id.startswith('TruncatedFourier'):
                str_name = 'Truncated Fourier'
            elif imaging_schemes[j].coding_id == 'Identity':
                str_name = 'FRH'
            elif imaging_schemes[j].coding_id == 'Greys':
                str_name = 'Count. Gray'
            elif imaging_schemes[j].coding_id.startswith('Learned'):
                str_name = 'Optimized'


            axs[0, i].plot(error_arr_bandlimited[:, i, j] / 10, marker='o', linestyle='-', label=str_name,
                           color=get_scheme_color(imaging_schemes[j].coding_id, 8, cw_tof=imaging_schemes[j].cw_tof,
                                                      constant_pulse_energy=imaging_schemes[j].constant_pulse_energy))
            axs[1, i].plot(error_arr_peaklimited[:, i, j] / 10, marker='o', linestyle='-', label=str_name,
                           color=get_scheme_color(imaging_schemes[j].coding_id, 8, cw_tof=imaging_schemes[j].cw_tof,
                                                      constant_pulse_energy=imaging_schemes[j].constant_pulse_energy))

            axs[0, i].set_xticks(np.arange(len(quants)))
            axs[0, i].set_xticklabels(quants)
            #axs[0, i].set_xlabel('Bit Size')


            axs[0, 0].set_ylabel('RMSE (cm)')

            axs[1, i].set_xticks(np.arange(len(quants)))
            axs[1, i].set_xticklabels(quants)
            axs[1, i].set_xlabel('Bit Size')

            axs[1, 0].set_ylabel('RMSE (cm)')

            axs[0, i].legend(fontsize=12)
            axs[1, i].legend(fontsize=12)

            if sbrs[i] <= 0.5:
                axs[0, i].set_title('Low SBR')
            else:
                axs[0, i].set_title('High SBR')

            axs[0, i].grid(True)
            axs[1, i].grid(True)


    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(f'bit_depth.svg', bbox_inches='tight')
    plt.show(block=True)


# Press the green