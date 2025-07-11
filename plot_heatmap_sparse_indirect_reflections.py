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

    sigma = 5
    sigmas = [5, 5]
    K_vals = [8, 10, 12]
    peak_factor = 0.030
    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, sigma, circ_shifted=True)

    params['meanBeta'] = 1e-4
    params['trials'] = 10
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 0.5
    depths = np.arange(1.0, params['dMax']-1.0, dSample)
    #depths = np.array([15.0])

    photon_count =  1000
    sbr = 0.1
    total_photons_indirects = [100, 300, 500, 700, 1000]
    #total_photons_indirects = [i/8 for i in total_photons_indirects]
    positions = [100, 200, 300, 400, 500]
    if peak_factor is not None:
        positions = [75, 150, 225, 300, 350]
    constant_pulse_energy = True

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

    error_all = np.zeros((len(total_photons_indirects), len(positions), len(K_vals), 4))
    for p in range(len(K_vals)):
        K = K_vals[p]
        if peak_factor is None:
            params['imaging_schemes'] = [
                # ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

                ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
                                    model=os.path.join('bandlimited_models', f'n{params["n_tbins"]}_k{K}_sigma{sigma}'),
                                    account_irf=True, h_irf=irf),

                ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=K, pulse_width=1, account_irf=True,
                                    h_irf=irf),

                ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=K, pulse_width=1, account_irf=True, h_irf=irf),
            ]
        else:
            peak_name = f"{peak_factor:.3f}".split(".")[-1]
            params['imaging_schemes'] = [
                #ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

                ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
                                    model=os.path.join('bandlimited_peak_models',
                                                       f'n{params["n_tbins"]}_k{K}_sigma{sigmas[0]}_peak015_counts1000'),
                                    h_irf=irf),
                ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
                                    model=os.path.join('bandlimited_peak_models',
                                                       f'n{params["n_tbins"]}_k{K}_sigma{sigmas[1]}_peak015_counts1000'),
                                    h_irf=irf),
                ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=K, pulse_width=1, account_irf=True,
                                    h_irf=irf, constant_pulse_energy=constant_pulse_energy),
                ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=K, pulse_width=1, account_irf=True, h_irf=irf,
                                    constant_pulse_energy=constant_pulse_energy),
            ]

        init_coding_list(n_tbins, params, t_domain=t_domain)
        imaging_schemes = params['imaging_schemes']

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

                    if 'Learned' in coding_scheme:
                        flag = False
                    else:
                        flag = constant_pulse_energy

                    incident, tmp_irf = light_obj.simulate_average_photons_sparse_indirect_reflections(photon_count,
                                                                                                       sbr,
                                                                                                       total_photons_indirect,
                                                                                                       position, depths,
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

                    error_all[k, l, p, i] = error_metrix['rmse'] / 10

    fig, ax = plt.subplots(len(K_vals), len(params['imaging_schemes']), figsize=(8, 8))
    for p in range(len(K_vals)):
        count = 0
        for i in range(len(params['imaging_schemes'])):
            imaging_schemes = params['imaging_schemes']
            str_name = ''
            if imaging_schemes[i].coding_id.startswith('TruncatedFourier'):
                str_name = 'Trunc. Fourier'
            elif imaging_schemes[i].coding_id == 'Identity':
                str_name = 'FRH'
            elif imaging_schemes[i].coding_id == 'Greys':
                str_name = 'Count. Gray'
            elif imaging_schemes[i].coding_id.startswith('Learned'):
                str_name = 'Opt. ' + r'$\sigma=' + str(sigmas[count]) + r'\Delta$'
                count += 1

            vmax = 100
            im = ax[p, i].imshow(error_all[:, :, p, i], vmin=3, vmax=vmax)
            ax[0, i].set_title(str_name)
            ax[p, 0].set_ylabel(f'K={K_vals[p]}')
            ax[p, i].tick_params(labelleft=False, labelbottom=False)

            for k in range(len(total_photons_indirects)):
                for l in range(len(positions)):
                    if error_all[k, l, p, i] > vmax-5:
                        text = ax[p, i].text(l, k, np.round(error_all[k, l, p, i], 1), ha='center', va='center',
                                             color='black', fontsize=6)
                    else:
                        text = ax[p, i].text(l, k, np.round(error_all[k, l, p, i], 1), ha='center', va='center',
                                             color='white', fontsize=6)

    cbar = ax[-1, -1].figure.colorbar(im, ax=ax[-1, -1], orientation='horizontal')
    cbar.set_ticks([])
    cbar.set_label("RMSE (cm)", fontsize=8)


    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # if peak_factor is not None:
    #    fig.savefig(f'sparse_indirect_reflections_peakk{peak_name}_fig.svg', bbox_inches='tight', dpi=1000)
    # else:
    #    fig.savefig(f'sparse_indirect_reflections_bandlimitedk{K}_fig.svg', bbox_inches='tight', dpi=3000)

    plt.show(block=True)

print()
print('YAYYY')
