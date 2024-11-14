from IPython.core import debugger

from spad_toflib.spad_tof_utils import normalize_measure_vals
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from plot_figures.plot_utils import *
import matplotlib.pyplot as plt

#matplotlib.use('Qt5Agg')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    breakpoint = debugger.set_trace

    # Press the green button in the gutter to run the script.
    if __name__ == '__main__':
        params = {}
        params['n_tbins'] = 2048
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
            #ImagingSystemParams('GrayTruncatedFourier', 'Gaussian', 'zncc', n_codes=20, pulse_width=1),
            ImagingSystemParams('KTapSinusoid', 'KTapSinusoid', 'zncc', ktaps=3),
            #ImagingSystemParams('Gated', 'Gaussian', 'zncc', pulse_width=1,
            #                    n_gates=10)
        ]

        params['meanBeta'] = 1e-4
        params['trials'] = 500
        params['freq_idx'] = [1]

        print(f'max depth: {params["dMax"]} meters')
        print()

        dSample = 1.0
        depths = np.arange(dSample, params['dMax'] - dSample, dSample)
        # depths = np.array([105.0])

        total_cycles = params['rep_freq'] * params['T']

        n_tbins = params['n_tbins']
        mean_beta = params['meanBeta']
        tau = params['rep_tau']
        depth_res = params['depth_res']
        t = params['T']
        trials = params['trials']
        (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
            (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))

        init_coding_list(n_tbins, depths, params, t_domain=t_domain)
        imaging_schemes = params['imaging_schemes']

        imaging_scheme = imaging_schemes[0]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        incident = light_obj.simulate_peak_photons(10, 2)

        coded_vals = coding_obj.encode(incident, trials).squeeze()

        decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * tbin_depth_res

        # imaging_scheme_pulsed = imaging_schemes[1]
        # coding_obj_pulsed = imaging_scheme_pulsed.coding_obj
        # coding_scheme_pulsed = imaging_scheme_pulsed.coding_id
        # light_obj_pulsed = imaging_scheme_pulsed.light_obj
        # light_source_pulsed = imaging_scheme_pulsed.light_id
        # rec_algo_pulsed = imaging_scheme_pulsed.rec_algo
        #
        # incident_pulsed = light_obj_pulsed.simulate_peak_photons(10, 2)
        #
        # coded_vals = coding_obj_pulsed.encode(incident_pulsed, trials).squeeze()


        inc = incident[18, 0, :]
        #modf = coding_obj.modfs[:, 0 ] * 20
        modf = light_obj.light_source * 20
        # inc_pulsed = incident_pulsed[18, 0, :]
        b_vals = normalize_measure_vals(coded_vals[100, 18, :])
        d_hat = int(decoded_depths[100, 18] / tbin_depth_res)
        z_correlations = coding_obj.zero_norm_corrfs
        corrs = coding_obj.correlations

        #demodfs = coding_obj.demodfs
        # fig, axs = plt.subplots()
        #
        # axs.set_xticks([])
        # axs.set_yticks([])
        # #axs.set_axis_off()
        # axs.spines['top'].set_visible(False)
        # axs.spines['right'].set_visible(False)
        # axs.spines['bottom'].set_position('zero')
        #
        #
        # axs.bar(np.arange(0, coding_obj.n_functions), b_vals, alpha=0.6,  edgecolor='black', linewidth=0.5, color='blue')
        # fig.savefig(os.path.join(save_folder, 'figure2e.svg'), bbox_inches='tight')
        # plt.show()
        # plt.close()
        #
        # fig, axs = plt.subplots()
        #
        # axs.set_xticks([])
        # axs.set_yticks([])
        # #axs.set_axis_off()
        # axs.spines['top'].set_visible(False)
        # axs.spines['right'].set_visible(False)
        #
        # axs.imshow(corrs.transpose(), aspect='auto', cmap=plt.cm.get_cmap('binary').reversed())
        # fig.savefig(os.path.join(save_folder, 'figure2f.svg'), bbox_inches='tight')
        # plt.show()
        # plt.close()

        fig, axs = plt.subplots()

        axs.set_xticks([])
        axs.set_yticks([])
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)

        #axs.plot(np.roll(modf * 2, 2), color='green')
        axs.plot(np.roll(modf, 2), color='blue')
        axs.plot(inc, color='lightblue')
        fig.savefig(os.path.join(save_folder, 'figure2b.svg'), bbox_inches='tight')
        plt.show()
        plt.close()
