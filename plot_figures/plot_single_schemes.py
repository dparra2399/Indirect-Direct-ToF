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
from matplotlib import rc


font = {'family': 'serif',
        'weight': 'bold',
        'size': 12}
rc('font', **font)


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
        ImagingSystemParams('Greys', 'Gaussian', 'ncc', pulse_width=1, n_bits=8,
                            account_irf=True, h_irf=gaussian_pulse(np.arange(params['n_tbins']), 0, 30, circ_shifted=True)),
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

    fig, axs = plt.subplots(1, 3, figsize=(10, 2))

    imaging_scheme = imaging_schemes[0]
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

    # Image on top
    img = np.repeat(coding_obj.decode_corrfs.transpose(), 100, axis=0)
    axs[1].imshow(img, cmap='gray', aspect='auto')

    for spine in axs[1].spines.values():
        spine.set_edgecolor('black')  # Set border color
        spine.set_linewidth(1)  # Set border thickness

    axs[1].set_xticks([])
    axs[1].set_yticks([])
    # Plot below the image
    # rect = patches.Rectangle((0, 5*100), img.shape[1], 96, linewidth=1, edgecolor='orange', facecolor='none')
    # axs[1].add_patch(rect)

    # rect = patches.Rectangle((0, 6*100+1), img.shape[1], 96, linewidth=1, edgecolor='green', facecolor='none')
    # axs[1].add_patch(rect)

    rect = patches.Rectangle((0, 7*100+1), img.shape[1], 96, linewidth=1, edgecolor='purple', facecolor='none')
    axs[1].add_patch(rect)

    #axs[2].plot(coding_obj.decode_corrfs[:, 5], linewidth=1.5, color='orange')
    #axs[2].plot(coding_obj.decode_corrfs[:, 6], linewidth=1.5, color='orange')
    axs[2].plot(coding_obj.decode_corrfs[:, 7], linewidth=1.5, color='purple')

    axs[2].set_ylim(-1, 1)
    # Hide ticks for clean visuals
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    #

    line1, = axs[0].plot(np.linspace(0, 1024, 1024), filtered_illum, color='blue', linewidth=1, label=r'$\Phi^{sig}$' + f'={int(np.sum(filtered_illum))}')
    for spine in axs[0].spines.values():
        spine.set_edgecolor('black')  # Set border color
        spine.set_linewidth(1)  # Set border thickness
    #ax1.set_ylabel('Intensity')
    axs[0].set_xlim(0, 1024)

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    #ax1.set_ylabel('Counts')


    if peak_factor is not None:
        axs[2].axhline(y=photon_count * peak_factor, color='red', linestyle='--', linewidth=2)

    for spine in axs[2].spines.values():
        spine.set_edgecolor('black')  # Set border color
        spine.set_linewidth(1)  # Set border thickness

    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[0].set_title('IRF h(t)')
    axs[1].set_title("Coding Matrix D'")
    axs[2].set_title("Last Three Rows of D'")

    #

#fig.tight_layout()
fig.subplots_adjust(wspace=0.05, hspace=0.05)
fig.savefig(f'Z:\\Research_Users\\David\\Learned Coding Functions Paper\\toy_example_figure.png', bbox_inches='tight', dpi=300)
plt.show(block=True)

print()
print('YAYYY')
