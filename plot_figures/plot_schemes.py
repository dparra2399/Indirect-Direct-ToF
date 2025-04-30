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

    params['imaging_schemes'] = [
        # ImagingSystemParams('Greys', 'Gaussian', 'ncc', pulse_width=1, n_bits=8,
        #                     account_irf=True,
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 1, circ_shifted=True)),
        #
        # ImagingSystemParams('Greys', 'Gaussian', 'ncc', pulse_width=1, n_bits=8,
        #                     account_irf=True, h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 30, circ_shifted=True)),
        # ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1, account_irf=False,
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 1, circ_shifted=True)),
        # ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=8),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', pulse_width=1, account_irf=True,
                            model=os.path.join('bandlimited_models', f'n1024_k8_sigma10'),
                            h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 10, circ_shifted=True)),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', pulse_width=1, account_irf=True,
                            model=os.path.join('bandlimited_models', f'n1024_k8_sigma20'),
                            h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 20, circ_shifted=True)),
        ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', pulse_width=1, account_irf=True,
                            model=os.path.join('bandlimited_models', f'n1024_k8_sigma30'),
                            h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 30, circ_shifted=True)),
        # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
        #                     model=os.path.join('bandlimited_peak_models', f'n1024_k8_sigma1_peak005_counts1000'),
        #                     h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 1, circ_shifted=True)),

    ]


    print(f'max depth: {params["dMax"]} meters')
    print()

    dSample = 0.5
    depths = np.arange(0, params['dMax'], dSample)
    # depths = np.array([105.0])

    photon_count =  1000
    sbr = 0.001
    peak_factor = None

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

    init_coding_list(n_tbins, depths, params, t_domain=t_domain)
    imaging_schemes = params['imaging_schemes']

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, len(imaging_schemes), figure=fig, hspace=0.5, wspace=0.1)

    for i in range(len(imaging_schemes)):
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
            peak_factor = None
            #pass

        incident = np.squeeze(light_obj.simulate_average_photons(photon_count, sbr, peak_factor=peak_factor))

        filtered_illum = np.roll(incident[0, :] - ((photon_count / sbr) / params['n_tbins']), int(n_tbins // 2))

        inner_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[i],
                                                    height_ratios=[3, 3, 1], hspace=0.17, wspace=0.05)

        # Image on top
        ax_img = fig.add_subplot(inner_gs[1])
        img = np.repeat(coding_obj.decode_corrfs.transpose(), 100, axis=0)
        ax_img.imshow(img, cmap='gray', aspect='auto')

        for spine in ax_img.spines.values():
            spine.set_edgecolor('black')  # Set border color
            spine.set_linewidth(1)  # Set border thickness

        ax_img.set_xticks([])
        ax_img.set_yticks([])
        # Plot below the image
        ax_plot = fig.add_subplot(inner_gs[2])

        ax_plot.plot(coding_obj.decode_corrfs[:, 6], linewidth=1.5, color='purple')
        ax_plot.plot(coding_obj.decode_corrfs[:, 1], linewidth=1.5, color='orange')

        rect = patches.Rectangle((0, 1*100), img.shape[1], 100, linewidth=2, edgecolor='orange', facecolor='none')
        ax_img.add_patch(rect)

        rect = patches.Rectangle((0, 6*100), img.shape[1], 100, linewidth=2, edgecolor='purple', facecolor='none')
        ax_img.add_patch(rect)

        # Hide ticks for clean visuals
        ax_plot.set_xticks([])
        ax_plot.set_yticks([])
        #
        ax1 = fig.add_subplot(inner_gs[0])  # Top-left plot

        line1, = ax1.plot(np.linspace(0, 1024, 1024), filtered_illum, color='blue', linewidth=2, label=r'$\Phi^{sig}$' + f'={int(np.sum(filtered_illum))}')
        for spine in ax1.spines.values():
            spine.set_edgecolor('black')  # Set border color
            spine.set_linewidth(1.5)  # Set border thickness
        ax1.set_ylabel('Intensity')
        ax1.set_xlabel('Time')

        ax1.set_xlim(0, 1024)

        ax1.set_xticks([])
        ax1.set_yticks([])
        #ax1.set_ylabel('Counts')


        if peak_factor is not None:
            ax1.axhline(y=photon_count * peak_factor, color='red', linestyle='--', linewidth=2)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        #

    #fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(f'Z:\\Research_Users\\David\\Learned Coding Functions Paper\\overview_schemes.png', bbox_inches='tight', dpi=300)
    plt.show(block=True)

print()
print('YAYYY')
