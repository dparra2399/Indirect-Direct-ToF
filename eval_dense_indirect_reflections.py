# Python imports
# Library imports
import time

from IPython.core import debugger
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from plot_figures.plot_utils import *
from matplotlib.lines import Line2D

font = {'family': 'serif',
        'weight': 'bold',
        'size': 18}

matplotlib.rc('font', **font)

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

    #irf = np.genfromtxt(r'C:\Users\Patron\PycharmProjects\Flimera-Processing\irfs\pulse_10mhz.csv', delimiter=',')
    irf=None

    sigma = 10
    peak_factor = 0.005
    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, sigma, circ_shifted=True)
    constant_pulse_energy = True


    if peak_factor is None:
        params['imaging_schemes'] = [
            #ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

            ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                model=os.path.join('bandlimited_models', f'n{params["n_tbins"]}_k8_sigma{sigma}'),
                                account_irf=True, h_irf=irf),

            ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf),

            ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1,  account_irf=True,
                                h_irf=irf),
        ]
    else:
        peak_name = f"{peak_factor:.3f}".split(".")[-1]
        params['imaging_schemes'] = [
            #ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

            ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
                                model=os.path.join('bandlimited_peak_models',
                                                   f'n{params["n_tbins"]}_k8_sigma{sigma}_peak015_counts1000'),
                                h_irf=irf),

            ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf, constant_pulse_energy=constant_pulse_energy),

            ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1, account_irf=True,
                                h_irf=irf, constant_pulse_energy=constant_pulse_energy),
        ]

    params['meanBeta'] = 1e-4
    params['trials'] = 5
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 0.5
    depths = np.arange(1.0, params['dMax']-1.0, dSample)
    #depths = np.array([15.0])

    photon_count =  1000
    sbr = 0.1
    decays = [80, 1000]
    if peak_factor is not None:
        decays = [80, 1000]
    amps = [0.05, 1]

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

    init_coding_list(n_tbins, params, t_domain=t_domain)
    imaging_schemes = params['imaging_schemes']

    fig, axs = plt.subplots(len(decays), len(amps), figsize=(10, 10))

    colors = [['purple', 'green'], ['blue', 'red']]
    colors2 = [['violet', 'lightgreen'], ['lightblue', 'orange']]

    for k in range(len(decays)):
        decay = decays[k]
        for l in range(len(amps)):
            amp = amps[l]

            handles = [Line2D([], [], linestyle='None', marker='',
                                          label=f'RMSE for depth range')]
            for i in range(len(imaging_schemes)):
                tic = time.perf_counter()
                imaging_scheme = imaging_schemes[i]
                coding_obj = imaging_scheme.coding_obj
                coding_scheme = imaging_scheme.coding_id
                light_obj = imaging_scheme.light_obj
                light_source = imaging_scheme.light_id
                rec_algo = imaging_scheme.rec_algo

                if 'Learned' in coding_scheme:
                    flag = False
                else:
                    flag = constant_pulse_energy

                incident, tmp_irf = light_obj.simulate_average_photons_dense_indirect_reflections(photon_count, sbr,
                                                                                                  decay,
                                                                                                  amp, depths,
                                                                                                  peak_factor=peak_factor,
                                                                                                  constant_pulse_energy=flag)

                print(f'\nphoton_count : {np.sum(incident[0, 0, :] - ((photon_count / sbr) / n_tbins))}')

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
                print_error_metrics(error_metrix, prefix=coding_scheme, K=coding_obj.n_functions)
                toc = time.perf_counter()

                if coding_scheme.startswith('TruncatedFourier'):
                    str_name = 'Trunc. Fourier'
                elif coding_scheme == 'Identity':
                    str_name = 'FRH'
                elif coding_scheme == 'Greys':
                    str_name = 'Count. Gray'
                elif coding_scheme.startswith('Learned'):
                    str_name = 'Optimized'


                if peak_factor is not None:
                    depth_choice = 20
                else:
                    depth_choice = 10

                if 'Learned' in coding_scheme and peak_factor is not None:
                    axs[k, l].plot(np.roll(incident[0, 0, :], n_tbins//2), color=colors2[k][l])
                elif 'Greys' in coding_scheme:
                    axs[k, l].plot(np.roll(incident[0, 0, :], n_tbins//2), color=colors[k][l])

                # if coding_scheme != 'Identity':
                #     axs[k, l].axvline(x=int(np.mean(decoded_depths[:, depth_choice]) / tbin_depth_res)-200,
                #                       color=get_scheme_color(coding_scheme, k, cw_tof=False,
                #                                 constant_pulse_energy=imaging_scheme.constant_pulse_energy),
                #                       linestyle='--', linewidth=2)
                #     handles.append(Line2D([], [], linestyle='None', marker='',
                #                           label=f'{str_name}: {error_metrix['rmse'] / 10:.2f}cm'))

            # legend = axs[k, l].legend(handles=handles, loc='upper right',handlelength=0,handletextpad=0, fontsize=5)
            # legend.get_texts()[0].set_color('black')
            # for tmp in range(len(imaging_schemes)):
            #     legend.get_texts()[tmp+1].set_color(color=get_scheme_color(imaging_schemes[tmp].coding_id, k,
            #                                                              cw_tof=imaging_schemes[tmp].cw_tof,
            #                                           constant_pulse_energy=imaging_schemes[tmp].constant_pulse_energy))

            axs[k, l].set_xticklabels([])
            axs[k, l].set_yticklabels([])

            axs[len(amps)-1, l].set_xlabel(f'Time (ns)')
            axs[len(amps)-1, l].set_xticks((np.linspace(0, n_tbins, 5)))
            axs[len(amps)-1, l].set_xticklabels((np.linspace(0, n_tbins-100, 5) * tbin_res * 1e9).astype(int))

            axs[k, 0].set_ylabel('Intensity')
            #ax_high.tick_params(axis='both', which='major', labelsize=6)



    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # if peak_factor is not None:
    #     fig.savefig('dense_indirect_reflections_peakk8_fig.svg', bbox_inches='tight', dpi=3000)
    # else:
    #     fig.savefig('dense_indirect_reflections_bandlimitedk8_fig.svg', bbox_inches='tight', dpi=3000)
    fig.savefig('tmp2.svg', bbox_inches='tight', dpi=1000)

    plt.show()


print()
print('YAYYY')