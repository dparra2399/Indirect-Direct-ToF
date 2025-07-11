# Python imports
# Library imports
import time

from IPython.core import debugger
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from plot_figures.plot_utils import *

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


    irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 10, circ_shifted=True)

    #irf = np.load(r'C:\Users\Patron\PycharmProjects\WISC-SinglePhoton3DData\system_irf\20190207_face_scanning_low_mu\ground_truth\irf_tres-8ps_tlen-17504ps.npy')
    params['imaging_schemes'] = [
        #ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf, constant_pulse_energy=True),
        ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf,
                            constant_pulse_energy=True),

        # ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=32),
        #ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1, account_irf=True,
        #                      h_irf=irf, constant_pulse_energy=True),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1,  account_irf=True,
                            h_irf=irf, constant_pulse_energy=True),
        ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=12, pulse_width=1, account_irf=True,
                            h_irf=irf, constant_pulse_energy=True),
        #
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_peak_models', 'n1024_k8_mae_fourier'),
        #                    account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Gaussian', 'zncc', pulse_width=1,
        #                    model=os.path.join('bandlimited_models', 'n2000_k8_sigma30'),
        #                    account_irf=True, h_irf=irf, quant=None),
        # ImagingSystemParams('LearnedImpulse', 'Gaussian', 'zncc', pulse_width=1,
        #                    model=os.path.join('bandlimited_models', 'n2000_k14_sigma30'),
        #                    account_irf=True, h_irf=irf, quant=None),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak015_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k10_sigma10_peak015_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k12_sigma10_peak015_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k14_sigma10_peak015_counts1000'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'n1024_k8_sigma30_photonstarved_v2'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'version_2'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'n1024_k8_sigma30'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'n1024_k8_sigma30'),
        #                     account_irf=True, h_irf=irf, fourier_coeff=1000),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'version_10'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                     model=os.path.join('bandlimited_models', 'version_9'),
        #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma1_peak015_counts1000'),
        #                     h_irf=irf, fourier_coeff=80),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
                            model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak005_counts1000'),
                            h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma1_peak030_counts1000'),
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 1, circ_shifted=True), quant=8),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma5_peak015_counts1000'),
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 5, circ_shifted=True)),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak015_counts1000'),
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 10, circ_shifted=True)),

        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                      model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak030_counts1000'),
        #                      account_irf=True, h_irf=irf),
        # # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        # #                     model=os.path.join('bandlimited_models', 'n2188_k8_spaddata'),
        # #                     account_irf=True, h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', pulse_width=1, account_irf=True,
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma5_peak030_counts1000'),
        #                     h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
        #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak015_counts1000'),
        #                     h_irf=irf),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
        #                      model=os.path.join('bandlimited_models', 'n1024_k10_sigma10_peak015_counts1000'),
        #                      account_irf=True, h_irf=irf),
        #ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf, constant_pulse_energy=True),
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

    ]


    params['meanBeta'] = 1e-4
    params['trials'] = 1000
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 0.5
    depths = np.arange(1.0, params['dMax']-1.0, dSample)
    # depths = np.array([105.0])

    photon_count =  1000
    sbr = 0.1
    peak_factor = 0.015

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

    for i in range(len(imaging_schemes)):
        tic = time.perf_counter()
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        if imaging_scheme.constant_pulse_energy:
            incident, tmp_irf = light_obj.simulate_constant_pulse_energy(photon_count, sbr, depths, peak_factor=peak_factor)
        else:
            incident, tmp_irf = light_obj.simulate_average_photons(photon_count, sbr, depths, peak_factor=peak_factor)

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
        print(f'{toc-tic:.6f} seconds')
        print()




print()
print('YAYYY')
