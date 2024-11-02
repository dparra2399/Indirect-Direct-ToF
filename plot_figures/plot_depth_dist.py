# Python imports
# Library imports
import time

from IPython.core import debugger
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from plot_figures.plot_utils import *
import matplotlib.pyplot as plt

#matplotlib.use('TkAgg')
breakpoint = debugger.set_trace

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    params = {}
    params['n_tbins'] = 2048
    #params['dMax'] = 5
    #params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 5 * 1e6
    params['rep_tau'] = 1. / params['rep_freq']
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['T'] = 0.1 #intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM

    #square_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_square_10mhz.npy')
    #pulse_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_pulse_10mhz.npy')

    pulse_width = 8e-9
    tbin_res = params['rep_tau'] / params['n_tbins']
    sigma = int(pulse_width / tbin_res)

    params['imaging_schemes'] = [
        #ImagingSystemParams('KTapSinusoid', 'KTapSinusoid', 'zncc', ktaps=3, cw_tof=False, split=False),
        ImagingSystemParams('KTapSinusoid', 'KTapSinusoid', 'zncc', ktaps=3, cw_tof=True, split=False),
        ImagingSystemParams('GrayTruncatedFourier', 'Gaussian', 'zncc', n_codes=20, pulse_width=1),
        #ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1),
    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 5000
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 1.0
    depths = np.arange(dSample, params['dMax']-dSample, dSample)
    # depths = np.array([105.0])

    #Do either average photon count
    photon_count = (10 ** 4)
    sbr = 0.1
    #Or peak photon count
    peak_photon_count = None
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

    all_errors = np.zeros(((params['trials']*depths.size), len(imaging_schemes)))
    n_codes = np.zeros((len(imaging_schemes)))
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
        all_errors[:, i] = np.reshape((decoded_depths - depths[np.newaxis, :]) * depth_res, (errors.size))
        n_codes[i] = coding_obj.n_functions
        error_metrix = calc_error_metrics(errors)
        print()
        print_error_metrics(error_metrix, prefix=coding_scheme)

        toc = time.perf_counter()
        print(f'{toc-tic:.6f} seconds')

    num_bins = 100
    bins = np.linspace(-500, 500, num_bins)
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    fig, axs = plt.subplots()
    axs.hist(all_errors[:, 0], bins, alpha=0.7, color=colors[0], density=True, label='I-ToF')
    axs.hist(all_errors[:, 1], bins, alpha=0.7, color=colors[1], density=True, label='d-ToF (Compressed)')
    #axs.hist(all_errors[:, 2], bins, alpha=0.7, color=colors[2])
    axs.set_ylabel('Depth Counts')
    axs.set_xlabel('Depth error')
    axs.set_title('Average Power')
    axs.legend()

    #axs.set_ylim(0, 10000)
    #fig.savefig('Z:\\Research_Users\\David\\paper figures\\figure2a.svg', bbox_inches='tight')
    fig.savefig('Z:\\Research_Users\\David\\paper figures\\ppt.png', bbox_inches='tight')
    plt.show()
    plt.close()

    fig,axs = plt.subplots()
    labels = ['i-ToF', 'd-ToF']
    x = [1, 2]
    axs.set_xticks(x)
    axs.set_xticklabels(labels)
    axs.bar(x, n_tbins / n_codes, color=colors)
    #fig.savefig('Z:\\Research_Users\\David\\paper figures\\figure2b.svg', bbox_inches='tight')
    fig.savefig('Z:\\Research_Users\\David\\paper figures\\ppt2.png', bbox_inches='tight')
    axs.set_ylabel('Compression Rate')
    axs.legend()
    #axs.set_xlabel('Depth error')
    plt.show()
    plt.close()
    print()
    print('YAYYY')
