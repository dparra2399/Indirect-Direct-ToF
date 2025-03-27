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

    params['meanBeta'] = 1e-4
    params['trials'] = 1
    params['freq_idx'] = [1]

    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 0.5
    depths = np.arange(0, params['dMax'], dSample)
    # depths = np.array([105.0])

    photon_count =  1000
    sbr = 1.0
    peak_factors = 0.010
    sigmas = [1, 5, 10]


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


    fig, ax = plt.subplots(4, len(sigmas), figsize=(8, 5))


    for j in range(len(sigmas)):
        sigma = sigmas[j]
        irf = gaussian_pulse(np.arange(params['n_tbins']), 0, sigma, circ_shifted=True)
        params['imaging_schemes'] = [
            ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

            ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', pulse_width=1, account_irf=True,
                                model=os.path.join('bandlimited_peak_models', f'n1024_k8_sigma{sigma}_peak030_counts1000'),
                                h_irf=irf),
            ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
                                model=os.path.join('bandlimited_peak_models', f'n1024_k8_sigma{sigma}_peak015_counts1000'),
                                h_irf=irf),
            ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
                                model=os.path.join('bandlimited_peak_models',
                                                   f'n1024_k8_sigma{sigma}_peak005_counts1000'),
                                h_irf=irf),

        ]
        ax[0][j].plot(np.roll(irf, int(n_tbins//2)), color='blue', linewidth=2)

        # ax[0].set_xticks([int(n_tbins // 2)])
        # ax[0].set_xticklabels([0])
        ax[0][j].set_xticks([])
        ax[0][j].set_yticks([])
        ax[0][j].spines['top'].set_visible(False)
        ax[0][j].spines['right'].set_visible(False)

        init_coding_list(n_tbins, depths, params, t_domain=t_domain)
        imaging_schemes = params['imaging_schemes']
        imaging_scheme_pulse = imaging_schemes[0]
        coding_obj_pulse = imaging_scheme_pulse.coding_obj
        light_obj_pulse = imaging_scheme_pulse.light_obj

        for i in range(1, len(imaging_schemes)):
            imaging_scheme = imaging_schemes[i]
            coding_obj = imaging_scheme.coding_obj
            coding_scheme = imaging_scheme.coding_id
            light_obj = imaging_scheme.light_obj
            light_source = imaging_scheme.light_id
            rec_algo = imaging_scheme.rec_algo

            try:
                #filename = imaging_schemes.model
                filename = imaging_scheme.model
                peak_factor = int(filename.split('_')[-2].split('peak')[-1]) * (1/1000)
            except:
                peak_factor = 1.0
                #pass

            incident = np.squeeze(light_obj.simulate_average_photons(photon_count, sbr, peak_factor=peak_factor))

            incident_pulse = np.squeeze(light_obj_pulse.simulate_average_photons(photon_count, sbr, peak_factor=peak_factor))

            delta_illum = np.roll(incident_pulse[0, :] - ((photon_count / sbr) / params['n_tbins']), int(n_tbins // 2))

            filtered_illum = np.roll(incident[0, :] - ((photon_count / sbr) / params['n_tbins']), int(n_tbins // 2))

            line1, = ax[i][j].plot(delta_illum, color='darkorange', linewidth=2, alpha=0.7, label=r'$\Phi^{sig}$' + f'={int(np.sum(delta_illum))}')
            line2, = ax[i][j].plot(filtered_illum, color='blue', linewidth=2, alpha=0.7, label=r'$\Phi^{sig}$' + f'={int(np.sum(filtered_illum))}')

            print(f'delta_illum: {np.sum(delta_illum)}')
            print(f'illum: {np.sum(filtered_illum)}')


            ax[i][j].set_xticks([])
            ax[-1][j].set_xticks([int(n_tbins)])
            ax[-1][j].set_xticklabels(['Time'])


            ax[i][j].set_yticks(np.append(np.array([0]),
                                                 np.append(np.round(np.linspace(0, photon_count * peak_factor, 4), 1),
                                                           np.array([photon_count * peak_factor]))))
            ax[i][j].set_yticklabels(np.append(np.array(['']),
                                                      np.append(np.round(np.linspace(0, photon_count * peak_factor, 4), 1),
                                                                np.array([f'{int(photon_count * peak_factor)}']))))

            if j > 0:
                ax[i][j].set_yticks([])
            else:
                ax[i][j].set_ylabel('Counts')


            ax[i][j].axhline(y=photon_count * peak_factor, color='red', linestyle='--', linewidth=2)
            ax[i][j].spines['top'].set_visible(False)
            ax[i][j].spines['right'].set_visible(False)

            block_legend1 = Line2D([0], [0], marker='s', color='w', markerfacecolor=line1.get_color(), markersize=6,
                                  label=line1.get_label())

            block_legend2 = Line2D([0], [0], marker='s', color='w', markerfacecolor=line2.get_color(), markersize=6,
                                  label=line2.get_label())

            ax[i][j].legend(handles=[block_legend1, block_legend2], loc='upper left', fontsize=9,handleheight=1.0, handlelength=0.1)

    #fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(f'Z:\\Research_Users\\David\\Learned Coding Functions Paper\\illum_peaks_grid.svg', bbox_inches='tight')
    plt.show(block=True)

print()
print('YAYYY')
