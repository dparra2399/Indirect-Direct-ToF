# Python imports
# Library imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import debugger

breakpoint = debugger.set_trace
from felipe_utils import tof_utils_felipe
from utils.coding_schemes_utils import init_coding_list
from spad_toflib import spad_tof_utils
from utils.coding_schemes_utils import ImagingSystemParams
from utils.file_utils import get_string_name

if __name__ == "__main__":

    params = {}
    params['n_tbins'] = 3000
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 5 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['rep_tau'] = 1. / params['rep_freq']
    params['T'] = 0.1  # intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM

    square_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_square_10mhz.npy')
    pulse_irf = np.load('/Users/Patron/PycharmProjects/Flimera-Processing/irf_pulse_10mhz.npy')

    params['imaging_schemes'] = [
        ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
                            duty=1. / 5.),
        ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                            duty=1. / 5.)
    ]

    params['meanBeta'] = 1e-4
    params['trials'] = 500
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()


    n_duty_lvls = 10
    (min_duty, max_duty) = (0.01, 0.50)

    peak_photon_count = 10
    ambient_count = 10 ** 4


    dSample = 1.0
    depths = np.arange(1.0, params['dMax'], dSample)

    n_duty_list = np.linspace(min_duty, max_duty, n_duty_lvls)


    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
    gt_tshifts = tof_utils_felipe.depth2time(depths)

    init_coding_list(params['n_tbins'], depths, params, t_domain=t_domain)

    imaging_schemes = params['imaging_schemes']
    trials = params['trials']
    results = np.zeros((len(imaging_schemes), n_duty_lvls))

    for i in range(len(imaging_schemes)):
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        for y in range(0, n_duty_lvls):
            duty = n_duty_list[y]

            coding_obj.set_duty(duty)
            coding_obj.set_coding_scheme(params['n_tbins'])

            light_obj.set_duty(duty)
            light_obj.set_light_source(light_obj.generate_source())

            incident = light_obj.simulate_peak_photons(peak_photon_count, ambient_count)

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

    fig, ax = plt.subplots()
    for j in range(len(imaging_schemes)):
        ax.plot(n_duty_list * 100, results[j, :])

        ax.scatter(x=n_duty_list * 100, y=results[j, :], label=get_string_name(imaging_schemes[j]))

    ax.legend()
    ax.set_xlabel('Peak Photon Count')
    ax.set_ylabel('MAE (mm)')
    ax.set_title(f'Peak Photons {peak_photon_count}, Amb. Photon per Bin {ambient_count / params["n_tbins"]}')
    ax.set_ylim(0, 500)
    ax.grid()
    plt.show()
    print('complete')
