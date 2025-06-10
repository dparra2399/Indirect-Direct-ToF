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
from matplotlib import rc
import matplotlib.gridspec as gridspec
import copy

font = {'family': 'serif',
        'size': 18}

rc('font', **font)
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
    fourier_coeffs = np.arange(10, 100, 10)
    sigmas = [30, 10]
    peak_factors = [None, 0.030]
    quants = [1, 2, 4, 8, 16, 32, 64]

    assert len(sigmas) == len(peak_factors), 'Must be the same!'

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


    error_arr_quants = np.zeros((len(sigmas), len(quants), len(sbrs), 3))
    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 10, circ_shifted=True)

    for q in range(len(sigmas)):
        sigma = sigmas[q]
        peak_factor = peak_factors[q]
        for j in range(len(quants)):
            quant = quants[j]
            params['imaging_schemes'] = [
                ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf),

                ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1,  account_irf=True,
                                    h_irf=irf, quant=quant),
            ]

            if peak_factor is None:
                params['imaging_schemes'].append(ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                 model=os.path.join('bandlimited_models', f'n1024_k8_sigma{sigma}'),
                                account_irf=True, h_irf=irf, quant=quant))
            else:
                peak_name = f"{peak_factor:.3f}".split(".")[-1]
                params['imaging_schemes'].append(ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                 model=os.path.join('bandlimited_peak_models', f'n1024_k8_sigma{sigma}_peak{peak_name}_counts1000'),
                                 account_irf=True, h_irf=irf, quant=quant))

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

                    coding_obj.update_tmp_irf(tmp_irf)
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

                    error_arr_quants[q, j, k, i] = error_metrix['rmse']

    imaging_schemes_tmp = copy.deepcopy(params['imaging_schemes'])

    error_arr_fourier = np.zeros((len(sigmas), fourier_coeffs.shape[0], len(sbrs), 2))
    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 10, circ_shifted=True)

    for q in range(len(sigmas)):
        sigma = sigmas[q]
        peak_factor = peak_factors[q]
        for j in range(fourier_coeffs.shape[0]):
            fourier_coeff = fourier_coeffs[j]
            params['imaging_schemes'] = [
                ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1, account_irf=True,
                                    h_irf=irf, fourier_coeff=fourier_coeff),
            ]

            if peak_factor is None:
                params['imaging_schemes'].append(ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                                                     model=os.path.join('bandlimited_models',
                                                                                        f'n1024_k8_sigma{sigma}'),
                                                                     account_irf=True, h_irf=irf, fourier_coeff=fourier_coeff))
            else:
                peak_name = f"{peak_factor:.3f}".split(".")[-1]
                params['imaging_schemes'].append(ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                                                     model=os.path.join('bandlimited_peak_models',
                                                                                        f'n1024_k8_sigma{sigma}_peak{peak_name}_counts1000'),
                                                                     account_irf=True, h_irf=irf, fourier_coeff=fourier_coeff))


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

                    coding_obj.update_tmp_irf(tmp_irf)
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

                    error_arr_fourier[q, j, k, i] = error_metrix['rmse']

    imaging_schemes_tmp2 = copy.deepcopy(params['imaging_schemes'])

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(len(sigmas), 2, figure=fig, hspace=0.05, wspace=0.08)

    for q in range(len(sigmas)):
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[q, 0],
                                                    width_ratios=[1, 1], hspace=0, wspace=0.05)

        ax_low = fig.add_subplot(inner_gs[0, 0])
        ax_high = fig.add_subplot(inner_gs[0, 1], sharey=ax_low)

        ax_low.grid(True)
        ax_high.grid(True)

        plt.setp(ax_high.get_yticklabels(), visible=False)
        ax_high.tick_params(left=False)


        if q != len(sigmas) - 1:
            ax_low.set_xticks(np.arange(len(quants)))
            ax_low.set_xticklabels(quants)
            ax_high.set_xticks(np.arange(len(quants)))
            ax_high.set_xticklabels(quants)
            plt.setp(ax_high.get_xticklabels(), visible=False)
            plt.setp(ax_low.get_xticklabels(), visible=False)
        else:
            ax_low.set_xticks(np.arange(len(quants)))
            ax_low.set_xticklabels(quants)
            ax_high.set_xticks(np.arange(len(quants)))
            ax_high.set_xticklabels(quants)
            ax_high.set_xlabel('# of Bits')

        if q == 0:
            ax_low.set_ylabel(r'$\sigma=30\Delta$' + '\n' + r'$\mathrm{p^{factor}}=\infty$' + '\n RMSE (cm)')
        elif q == 1:
            ax_low.set_ylabel(r'$\sigma=10\Delta$' + '\n' + r'$\mathrm{p^{factor}}=0.015$' + '\n RMSE (cm)')
        #ax_low.set_ylabel('RMSE (cm)')

        for label in ax_low.get_yticklabels():
            label.set_rotation(90)

        for label in ax_high.get_yticklabels():
            label.set_rotation(90)

        for l in range(len(imaging_schemes_tmp)):

            if imaging_schemes_tmp[l].coding_id.startswith('TruncatedFourier'):
                str_name = 'Trunc. Fourier'
            elif imaging_schemes_tmp[l].coding_id == 'Identity':
                str_name = 'FRH'
            elif imaging_schemes_tmp[l].coding_id == 'Greys':
                str_name = 'Count. Gray'
            elif imaging_schemes_tmp[l].coding_id.startswith('Learned'):
                str_name = 'Optimized'


            ax_low.plot(error_arr_quants[q, :, 0, l] / 10, marker='o', linestyle='-', label=str_name,
                           color=get_scheme_color(imaging_schemes_tmp[l].coding_id, 8, cw_tof=imaging_schemes_tmp[l].cw_tof,
                                                      constant_pulse_energy=imaging_schemes_tmp[l].constant_pulse_energy))

            ax_high.plot(error_arr_quants[q, :, 1, l] / 10, marker='o', linestyle='-', label=str_name,
                           color=get_scheme_color(imaging_schemes_tmp[l].coding_id, 8, cw_tof=imaging_schemes_tmp[l].cw_tof,
                                                      constant_pulse_energy=imaging_schemes_tmp[l].constant_pulse_energy))

        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[q, 1],
                                                    width_ratios=[1, 1], hspace=0, wspace=0.05)

        ax_low2 = fig.add_subplot(inner_gs[0, 0])
        ax_high2 = fig.add_subplot(inner_gs[0, 1], sharey=ax_low2)

        ax_low2.grid(True)
        ax_high2.grid(True)

        plt.setp(ax_high2.get_yticklabels(), visible=False)
        ax_high2.tick_params(left=False)

        if q != len(sigmas) - 1:
            ax_low2.set_xticks(np.arange(fourier_coeffs.shape[0]))
            ax_low2.set_xticklabels([
                str(fourier_coeffs[i]) if i % 2 == 0 else ''
                for i in range(fourier_coeffs.shape[0])
            ])
            ax_high2.set_xticks(np.arange(fourier_coeffs.shape[0]))
            ax_high2.set_xticklabels([
                str(fourier_coeffs[i]) if i % 2 == 0 else ''
                for i in range(fourier_coeffs.shape[0])
            ])
            plt.setp(ax_high2.get_xticklabels(), visible=False)
            plt.setp(ax_low2.get_xticklabels(), visible=False)
        else:
            ax_low2.set_xticks(np.arange(fourier_coeffs.shape[0]))
            ax_low2.set_xticklabels([
                str(fourier_coeffs[i]) if i % 2 == 0 else ''
                for i in range(fourier_coeffs.shape[0])
            ])

            ax_high2.set_xticks(np.arange(fourier_coeffs.shape[0]))
            ax_high2.set_xticklabels([
                str(fourier_coeffs[i]) if i % 2 == 0 else ''
                for i in range(fourier_coeffs.shape[0])
            ])
            ax_high2.set_xlabel('# of Fourier Coeff.')

        for label in ax_low2.get_yticklabels():
            label.set_rotation(90)

        for label in ax_high2.get_yticklabels():
            label.set_rotation(90)

        for l in range(len(imaging_schemes_tmp2)):

            if imaging_schemes_tmp2[l].coding_id.startswith('TruncatedFourier'):
                str_name = 'Trunc. Fourier'
            elif imaging_schemes_tmp2[l].coding_id == 'Identity':
                str_name = 'FRH'
            elif imaging_schemes_tmp2[l].coding_id == 'Greys':
                str_name = 'Count. Gray'
            elif imaging_schemes_tmp2[l].coding_id.startswith('Learned'):
                str_name = 'Optimized'


            ax_low2.plot(error_arr_fourier[q, :, 0, l] / 10, marker='o', linestyle='-', label=str_name,
                           color=get_scheme_color(imaging_schemes_tmp2[l].coding_id, 8, cw_tof=imaging_schemes_tmp2[l].cw_tof,
                                                      constant_pulse_energy=imaging_schemes_tmp2[l].constant_pulse_energy))

            ax_high2.plot(error_arr_fourier[q, :, 1, l] / 10, marker='o', linestyle='-', label=str_name,
                           color=get_scheme_color(imaging_schemes_tmp2[l].coding_id, 8, cw_tof=imaging_schemes_tmp2[l].cw_tof,
                                                      constant_pulse_energy=imaging_schemes_tmp2[l].constant_pulse_energy))

            if q == 0:
                ax_low.set_title('Low SNR')
                ax_high.set_title('High SNR')
                ax_low2.set_title('Low SNR')
                ax_high2.set_title('High SNR')

                ax_low.legend(fontsize=12)
                ax_high.legend(fontsize=12)
                ax_low2.legend(fontsize=12)
                ax_high2.legend(fontsize=12)


    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(f'bit_depth_fourier_figure.svg', bbox_inches='tight')
    plt.show(block=True)
