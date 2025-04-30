from IPython.core import debugger
from IPython.core.pylabtools import figsize

from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from utils.file_utils import get_string_name
from plot_figures.plot_utils import *
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
import matplotlib.pyplot as plt
import matplotlib.patches as patches


font = {'family': 'serif',
        'size': 7
        }
matplotlib.rc('font', **font)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    breakpoint = debugger.set_trace

    # Press the green button in the gutter to run the script.
    if __name__ == '__main__':
        params = {}
        params['n_tbins'] = 1024
        # params['dMax'] = 5
        # params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
        params['rep_freq'] = 5 * 1e6
        params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
        params['rep_tau'] = 1. / params['rep_freq']
        params['T'] = 0.1  # intergration time [used for binomial]
        params['depth_res'] = 1000  ##Conver to MM

        sigma = 1
        irf = gaussian_pulse(np.arange(params['n_tbins']), 0, sigma, circ_shifted=True)
        params['imaging_schemes'] = [
            ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf),

            # ImagingSystemParams('Gated', 'Gaussian', 'linear', pulse_width=1, n_gates=32),
            ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1, account_irf=False,
                                h_irf=irf),
            #
            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', model=os.path.join('bandlimited_peak_models', 'n1024_k8_mae_fourier'),
            #                    account_irf=True, h_irf=irf),
            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
            #                     model=os.path.join('bandlimited_models', 'n1024_k8_sigma30'),
            #                     account_irf=True, h_irf=irf),
            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
            #                     model=os.path.join('bandlimited_models', f'n1024_k8_sigma{sigma}'),
            #                     account_irf=True, h_irf=irf),
            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
            #                     model=os.path.join('bandlimited_models', 'version_2'),
            #                     account_irf=True, h_irf=irf),
            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
            #                     model=os.path.join('bandlimited_models', 'n1024_k8_sigma10'),
            #                     account_irf=True, h_irf=irf),
            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
            #                     model=os.path.join('bandlimited_models', 'version_10'),
            #                     account_irf=True, h_irf=irf),
            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
            #                     model=os.path.join('bandlimited_models', 'version_9'),
            #                     account_irf=True, h_irf=irf),

            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
            #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak005_counts1000'),
            #                     account_irf=True, h_irf=irf),
            # # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
            # #                     model=os.path.join('bandlimited_models', 'n2188_k8_spaddata'),
            # #                     account_irf=True, h_irf=irf),
            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', pulse_width=1, account_irf=True,
            #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma5_peak030_counts1000'),
            #                     h_irf=irf),
            # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc', account_irf=True,
            #                     model=os.path.join('bandlimited_peak_models', 'n1024_k8_sigma10_peak015_counts1000'),
            #                     h_irf=irf),

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

        fig, axs = plt.subplots(nrows=len(imaging_schemes), ncols=4, figsize=(8, 4))

        axs[0][1].set_title('Input Illumination f(t)')
        axs[0][3].set_title('Rows 2 and 7 of D')
        axs[0][2].set_title('Coding Matrix D')
        for i in range(len(imaging_schemes)):
            imaging_scheme = imaging_schemes[i]
            coding_obj = imaging_scheme.coding_obj
            coding_scheme = imaging_scheme.coding_id
            light_obj = imaging_scheme.light_obj
            light_source = imaging_scheme.light_id
            rec_algo = imaging_scheme.rec_algo

            axs[i][1].set_xticks([])
            axs[i][1].set_yticks([])
            axs[i][2].set_xticks([])
            axs[i][2].set_yticks([])
            axs[i][3].set_xticks([])
            axs[i][3].set_yticks([])
            axs[i][1].spines['top'].set_visible(False)
            axs[i][1].spines['right'].set_visible(False)

            axs[i][1].set_ylabel('Intensity')
            axs[i][1].set_xlabel('Time')
            axs[i][3].set_xlabel('Time')

            axs[i][3].plot(coding_obj.decode_corrfs[:, 6], color='orange')
            axs[i][3].plot(coding_obj.decode_corrfs[:, 2], color='purple')
            #axs[i][2].set_ylim(-1, 1)
            tmp = coding_obj.decode_corrfs.transpose()
            #tmp[4, :] = 0

            img = np.repeat(coding_obj.decode_corrfs.transpose(), 100, axis=0)
            axs[i][2].imshow(img, cmap='gray', aspect='auto')

            rect = patches.Rectangle((0, 1 * 100), img.shape[1], 100, linewidth=2, edgecolor='orange', facecolor='none')
            axs[i][2].add_patch(rect)

            rect = patches.Rectangle((0, 6 * 100), img.shape[1], 100, linewidth=2, edgecolor='purple', facecolor='none')
            axs[i][2].add_patch(rect)

            axs[i][2].set_axis_off()

            first_zero_index = np.where(light_obj.filtered_light_source == 0)[0]
            axs[i][1].plot(np.roll(light_obj.filtered_light_source, int(n_tbins // 2)), color='blue')
            axs[i][1].set_xticks([])
            axs[i][0].set_axis_off()
            axs[i][0].text(0.0, 0.5, f'{get_string_name(imaging_scheme)}')
    #fig.text(0.04, 0.25, 'Hamiltonian', va='center', rotation='vertical', fontsize=7)

    fig.suptitle(f'Coding Scheme with IRF width $\sigma$={sigma}$\Delta$', fontsize=12,  fontweight="bold")
    fig.tight_layout()  # Adjust the rect to make space for the common labels


    for i in range(len(axs) - 1):
        row_bottom = axs[i, 0].get_position().y0

        # Add horizontal line
        fig.add_artist(plt.Line2D(
            [0, 1], [row_bottom-0.04, row_bottom-0.04], transform=fig.transFigure,
            color='black', linewidth=1
        ))

    row_top = axs[0, 0].get_position().y1

    # Add horizontal line
    fig.add_artist(plt.Line2D(
        [0, 1], [row_top+0.08, row_top+0.08], transform=fig.transFigure,
        color='black', linewidth=1
    ))

    for j in range(len(axs[0]) - 1):
        col_left = axs[0, j].get_position().x1
        if j > 0:
            col_left += 0.02
        else:
            col_left -= 0.02
        fig.add_artist(plt.Line2D(
            [col_left, col_left], [0, 0.935], transform=fig.transFigure,
            color='black', linewidth=1
        ))

    fig.savefig(f'Z:\\Research_Users\\David\\Learned Coding Functions Paper\\supp_sigma{sigma}_coding.png', bbox_inches='tight', dpi=300)
    plt.show()

