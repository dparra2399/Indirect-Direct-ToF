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
        'size': 12}

matplotlib.rc('font', **font)

#matplotlib.use('QTkAgg')
breakpoint = debugger.set_trace

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    params = {}
    params['n_tbins'] = 2000
    #params['dMax'] = 5
    #params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 5 * 1e6
    params['rep_tau'] = 1. / params['rep_freq']
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['T'] = 0.1 #intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM

    #irf = np.genfromtxt(r'C:\Users\Patron\PycharmProjects\Flimera-Processing\irfs\pulse_10mhz.csv', delimiter=',')
    irf=None

    sigma = 30
    K = 10
    peak_factor = None #0.015
    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, sigma, circ_shifted=True)


    if peak_factor is None:
        params['imaging_schemes'] = [
            #ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

            ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                model=os.path.join('bandlimited_models', f'n{params["n_tbins"]}_k{K}_sigma{sigma}'),
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
                                                   f'n{params["n_tbins"]}_k{K}_sigma{sigma}_peak015_counts1000'),
                                h_irf=irf),

            ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf),

            ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1, account_irf=True,
                                h_irf=irf),
        ]

    params['meanBeta'] = 1e-4
    params['trials'] = 5000
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 0.5
    depths = np.arange(1.0, params['dMax']-1.0, dSample)
    #depths = np.array([15.0])

    photon_count =  1000
    sbr = 0.1
    total_photons_indirects = 2 * [100, 300, 500, 700, 1000]
    if peak_factor is not None:
        total_photons_indirects = [25, 75, 125, 175, 250]
    positions = [100, 200, 300, 400, 500]

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

    fig, axs = plt.subplots(len(total_photons_indirects), len(positions), figsize=(10, 10))
    for k in range(len(total_photons_indirects)):
        total_photons_indirect = total_photons_indirects[k]
        for l in range(len(positions)):
            position = positions[l]

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


                incident, tmp_irf = light_obj.simulate_average_photons_sparse_indirect_reflections(photon_count, sbr,
                                                                                            total_photons_indirect,
                                                                                            position, depths, peak_factor=peak_factor)

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
                    axs[k, l].plot(incident[depth_choice, 0, :], color='orange', alpha=0.6)
                elif 'Greys' in coding_scheme:
                    axs[k, l].plot(incident[depth_choice, 0, 200:1400], color='blue', alpha=0.6)

                if coding_scheme != 'Identity':
                    axs[k, l].axvline(x=int(np.mean(decoded_depths[:, depth_choice]) / tbin_depth_res)-200,
                                      color=get_scheme_color(coding_scheme, k, cw_tof=False,
                                                constant_pulse_energy=imaging_scheme.constant_pulse_energy),
                                      linestyle='--', linewidth=2)
                    handles.append(Line2D([], [], linestyle='None', marker='',
                                          label=f'{str_name}: {error_metrix['rmse'] / 10:.2f}cm'))

            legend = axs[k, l].legend(handles=handles, loc='upper right',handlelength=0,handletextpad=0, fontsize=5)
            legend.get_texts()[0].set_color('black')
            for tmp in range(len(imaging_schemes)):
                legend.get_texts()[tmp+1].set_color(color=get_scheme_color(imaging_schemes[tmp].coding_id, k,
                                                                         cw_tof=imaging_schemes[tmp].cw_tof,
                                                      constant_pulse_energy=imaging_schemes[tmp].constant_pulse_energy))

            axs[k, l].set_xticklabels([])
            axs[k, l].set_yticklabels([])

            axs[len(positions)-1, l].set_xlabel(f'Time (ns)')
            axs[len(positions)-1, l].set_xticks((np.linspace(0, 1100, 5)))
            axs[len(positions)-1, l].set_xticklabels((np.linspace(200, 1300, 5) * tbin_res * 1e9).astype(int))

            axs[k, 0].set_ylabel('Intensity')
            #ax_high.tick_params(axis='both', which='major', labelsize=6)



    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    if peak_factor is not None:
        fig.savefig(f'sparse_indirect_reflections_peakk{K}_fig.svg', bbox_inches='tight', dpi=3000)
    else:
        fig.savefig(f'sparse_indirect_reflections_bandlimitedk{K}_fig.svg', bbox_inches='tight', dpi=3000)

    plt.show()


print()
print('YAYYY')
