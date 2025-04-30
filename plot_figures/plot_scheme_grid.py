# Python imports
# Library imports
import time

import numpy as np
from IPython.core import debugger
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from plot_figures.plot_utils import *
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches



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
    peak_factor = None
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

    num_rows, num_cols = 3, 3
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=0.05, wspace=0.05)

    for i in range(len(sigmas)):
        sigma = sigmas[i]

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

        init_coding_list(n_tbins, depths, params, t_domain=t_domain)
        imaging_schemes = params['imaging_schemes']
        imaging_scheme_pulse = imaging_schemes[0]
        coding_obj_pulse = imaging_scheme_pulse.coding_obj
        light_obj_pulse = imaging_scheme_pulse.light_obj

        for j in range(len(imaging_schemes)-1):
            imaging_scheme = imaging_schemes[j+1]
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
            #filtered_illum = np.roll(light_obj.light_source, int(n_tbins // 2))

            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[j, i],
                                                        height_ratios=[1, 1], hspace=0, wspace=0)

            # Image on top
            ax_img = fig.add_subplot(inner_gs[0, 0])
            img = np.repeat(coding_obj.decode_corrfs.transpose(), 100, axis=0)
            ax_img.imshow(img, cmap='gray', aspect='auto')



            ax_img.axis('off')  # Hide axis for image

            # Plot below the image
            ax_plot = fig.add_subplot(inner_gs[0, 1])

            if True:
                ax_plot.plot(coding_obj.decode_corrfs[:, 1], linewidth=1, color='orange')
                ax_plot.plot(coding_obj.decode_corrfs[:, 6], linewidth=1, color='purple')

                rect = patches.Rectangle((0, 1*100), img.shape[1], 100, linewidth=1, edgecolor='orange', facecolor='none')
                ax_img.add_patch(rect)

                rect = patches.Rectangle((0, 6*100), img.shape[1], 100, linewidth=1, edgecolor='purple', facecolor='none')
                ax_img.add_patch(rect)
            else:
                ax_plot.plot(coding_obj.decode_corrfs, linewidth=1)

            # Hide ticks for clean visuals
            ax_plot.set_xticks([])
            ax_plot.set_yticks([])
            #
            ax1 = fig.add_subplot(inner_gs[1, :])  # Top-left plot

            line2, = ax1.plot(np.linspace(0, 1024, 1024), delta_illum, color='deepskyblue', linewidth=1, label=r'$\Phi^{sig}$' + f'={int(np.sum(delta_illum))}')
            line1, = ax1.plot(np.linspace(0, 1024, 1024), filtered_illum, color='blue', linewidth=1, label=r'$\Phi^{sig}$' + f'={int(np.sum(filtered_illum))}')

            ax1.set_xlim(0, 1024)

            #
            #

            #
            ax1.set_xticks([])
            if j == 2:
                ax1.set_xticks([int(n_tbins)])
                ax1.set_xticklabels(['Time'])


            ax1.set_yticks(np.append(np.array([0]),
                                                 np.append((np.linspace(0, photon_count * peak_factor, 4)).astype(int),
                                                           np.array([photon_count * peak_factor]))))
            ax1.set_yticklabels(np.append(np.array(['']),
                                                      np.append((np.linspace(0, photon_count * peak_factor, 4)).astype(int),
                                                                np.array([f'{int(photon_count * peak_factor)}']))))

            if i > 0:
                ax1.set_yticks([])
            else:
                ax1.set_ylabel('Counts')


            ax1.axhline(y=photon_count * peak_factor, color='red', linestyle='--', linewidth=2)
            #ax1.spines['top'].set_visible(False)
            #ax1.spines['right'].set_visible(False)
            #
            block_legend1 = Line2D([0], [0], marker='s', color='w', markerfacecolor=line1.get_color(), markersize=6,
                                    label=line1.get_label())

            block_legend2 = Line2D([0], [0], marker='s', color='w', markerfacecolor=line2.get_color(), markersize=6,
                                  label=line2.get_label())

            ax1.legend(handles=[block_legend1, block_legend2], loc='upper left', fontsize=9,handleheight=1.0, handlelength=0.1)

    #fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(f'Z:\\Research_Users\\David\\Learned Coding Functions Paper\\illum_coding_grid.svg', bbox_inches='tight')
    plt.show(block=True)

print()
print('YAYYY')
