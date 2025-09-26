# Python imports
# Library imports
import time

from IPython.core import debugger

from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from plot_figures.plot_utils import *
from scipy.interpolate import interp1d

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
    params['depth_res'] = 1000  ##Conver to MM

    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 10, circ_shifted=True)

    gated = True
    binomial = True
    debug = False
    split_measurements = False
    after = True
    params['imaging_schemes'] = [
        ImagingSystemParams('Gated', 'Gaussian', 'zncc', pulse_width=50, n_gates=8, h_irf=irf, account_irf=True,
                            gated=gated, binomial=binomial, split_measurements=split_measurements, cw_tof=after),
        ImagingSystemParams('Gated', 'Gaussian', 'zncc', pulse_width=100, n_gates=4, h_irf=irf, account_irf=True,
                            gated=gated, binomial=binomial, split_measurements=split_measurements, cw_tof=after),
        ImagingSystemParams('Gated', 'Gaussian', 'zncc', pulse_width=120, n_gates=3, h_irf=irf, account_irf=True,
                            gated=gated, binomial=binomial, split_measurements=split_measurements, cw_tof=after),
        ImagingSystemParams('Greys', 'Gaussian', 'zncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf,
                             gated=gated, binomial=binomial, split_measurements=split_measurements, cw_tof=after),

        #ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', pulse_width=1, n_codes=8, h_irf=irf, account_irf=True,
        #                    gated=False, binomial=False),
        ImagingSystemParams('HamiltonianK3', 'HamiltonianK3', 'zncc', duty=1/6, h_irf=irf, account_irf=True,
                            gated=gated, binomial=binomial, split_measurements=split_measurements, cw_tof=after),
        ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc', duty=1/12, h_irf=irf, account_irf=True,
                            gated=gated, binomial=binomial, split_measurements=split_measurements, cw_tof=after),
        ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc', duty=1 / 30, h_irf=irf,
                            account_irf=True,
                            gated=gated, binomial=binomial, split_measurements=split_measurements, cw_tof=after),
        #ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                    model=os.path.join('bandlimited_binary_models', f'version_1'),
        #                    gated=gated, account_irf=True, h_irf=irf),

    ]


    params['meanBeta'] = 1e-4
    params['trials'] = 1000
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 1.0
    depths = np.arange(5.0, params['dMax']-5.0, dSample)
    # depths = np.array([105.0])

    photon_count = 300
    sbr = 0.1
    peak_factor = None #0.030
    laser_cycles = 2 * 1e6
    integration_time = 0.1 #in milliseconds


    n_tbins = params['n_tbins']
    mean_beta = params['meanBeta']
    tau = params['rep_tau']
    depth_res = params['depth_res']
    trials = params['trials']
    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))

    print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')
    print()

    init_coding_list(n_tbins, params, t_domain=t_domain)
    imaging_schemes = params['imaging_schemes']

    for i in range(len(imaging_schemes)):
        tic = time.perf_counter()
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        if imaging_scheme.constant_pulse_energy and peak_factor is not None:
            incident, tmp_irf = light_obj.simulate_constant_pulse_energy(photon_count, sbr, depths,
                                                                         peak_factor=peak_factor)
        else:
            incident, tmp_irf = light_obj.simulate_average_photons(photon_count, sbr, depths,
                                                                   peak_factor=peak_factor, t=integration_time)

        #print(f'\nphoton_count : {np.sum(incident[0, 0, :] - ((photon_count / sbr) / n_tbins))}')

        coded_vals = coding_obj.encode(incident, trials, laser_cycles, debug=debug).squeeze()

        if light_source in 'Gaussian':
            coding_obj.update_tmp_irf(tmp_irf)
            coding_obj.update_C_derived_params()


        # fig, axs = plt.subplots(1, 2, sharex=True)
        # axs[0].imshow(np.repeat(coding_obj.decode_corrfs.transpose(), 100, axis=0), aspect='auto')
        # axs[1].plot(coding_obj.decode_corrfs)
        # plt.show()
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
        print(f'{toc-tic:.6f} seconds')
        print()




print()
print('YAYYY')
