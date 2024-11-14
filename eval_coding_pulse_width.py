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

if __name__ == "__main__":

    params = {}
    params['n_tbins'] = 1024
    # params['dMax'] = 5
    # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
    params['rep_freq'] = 5 * 1e6
    params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
    params['rep_tau'] = 1. / params['rep_freq']
    params['T'] = 0.1  # intergration time [used for binomial]
    params['depth_res'] = 1000  ##Conver to MM


    pulse_width = 8e-9
    tbin_res = params['rep_tau'] / params['n_tbins']
    sigma = int(pulse_width / tbin_res)

    params['imaging_schemes'] = [
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=4, pulse_width=sigma),
        ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                            duty=1. / 4, freq_window=0.20)
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
    peak_photon_count = 20
    ambient_count = 5

    (min_pulse_width, max_pulse_width) = (1, 100)
    (min_duty_cycle, max_duty_cycle) = (0.01, 0.50)
    sample = 20

    pw_list = np.linspace(min_pulse_width, max_pulse_width, sample)
    duty_list = np.linspace(min_duty_cycle, max_duty_cycle, sample)


    trials = params['trials']
    results = np.zeros((len(params['imaging_schemes']), pw_list.shape[0]))

    for y in range(0, pw_list.shape[0]):
        sigma = int(pw_list[y])
        duty = duty_list[y]
        params['imaging_schemes'] = [
            ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=sigma),
            #ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=4, pulse_width=sigma),
            #ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
            #                    duty=duty, freq_window=0.20)
        ]
        (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
            (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
        init_coding_list(params['n_tbins'], depths, params, t_domain=t_domain)
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

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    for j in range(len(imaging_schemes)):
        ax1.plot(pw_list, results[j, :])
        ax1.scatter(x=pw_list, y=results[j, :], label=imaging_schemes[j].coding_id)

    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xticks(np.round(pw_list * 100, 3))
    # ax2.set_xticklabels(duty_list)

    ax2.set_xlabel(r"DUTY CYCLE")

    ax1.legend()
    ax1.set_xlabel('SIGMA')
    ax1.set_ylabel('MAE (mm)')
    #ax1.set_ylim(0, 300)
    plt.show()
    fig.savefig('Z:\\Research_Users\\David\\paper figures\\supfigure4a.svg', bbox_inches='tight')
    print('complete')
