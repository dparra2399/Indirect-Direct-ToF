# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger

breakpoint = debugger.set_trace
from felipe_utils.felipe_impulse_utils import tof_utils_felipe
from utils.coding_schemes_utils import init_coding_list
from spad_toflib import spad_tof_utils
from utils.coding_schemes_utils import ImagingSystemParams

if __name__ == "__main__":

    params = {}
    params['n_tbins'] = 1000000
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 5 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['rep_tau'] = 1. / params['rep_freq']
    params['T'] = 0.1  # intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM

    square_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_square_10mhz.npy')
    pulse_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_pulse_10mhz.npy')

    pulse_width = 8e-9
    tbin_res = params['rep_tau'] / params['n_tbins']
    sigma = int(pulse_width / tbin_res)

    params['imaging_schemes'] = [
        ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
                            duty=1. / 5.),
        ImagingSystemParams('HamiltonianK3', 'HamiltonianK3', 'zncc',
                            duty=1. / 5.),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=5),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=sigma)
    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 500
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 1.0
    depths = np.arange(0.1, params['dMax'], dSample)
    # depths = np.array([105.0])

    #Do either average photon count
    photon_count = (10 ** 6)
    sbr = 1
    #Or peak photon count
    peak_photon_count = 8
    ambient_count = 10

    (min_time_bins, max_time_bins) = (1000, 10_000)
    time_bin_sample = 1000
    n_tbins_list = np.arange(min_time_bins, max_time_bins, time_bin_sample)

    trials = params['trials']
    results = np.zeros((len(params['imaging_schemes']), n_tbins_list.shape[0]))

    for y in range(0, n_tbins_list.shape[0]):
        n_tbins = n_tbins_list[y]
        (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
            (tof_utils_felipe.calc_tof_domain_params(n_tbins, rep_tau=params['rep_tau']))
        init_coding_list(n_tbins, depths, params, t_domain=t_domain)
        imaging_schemes = params['imaging_schemes']
        for i in range(len(imaging_schemes)):
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

            imaging_scheme.mean_absolute_error = spad_tof_utils.compute_metrics(depths, decoded_depths) * params[
                'depth_res']
            results[i, y] = imaging_scheme.mean_absolute_error

    for j in range(len(imaging_schemes)):
        plt.plot(n_tbins_list, results[j, :])
        plt.scatter(x=n_tbins_list, y=results[j, :], label=imaging_schemes[j].coding_id)

    plt.legend()
    plt.xlabel('Number of Time Bins')
    plt.ylabel('MAE (mm)')
    plt.grid()
    plt.show()
    print('complete')
