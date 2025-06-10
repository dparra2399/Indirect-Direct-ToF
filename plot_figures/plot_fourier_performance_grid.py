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

    filename = r'../data/results/fourier_performance/fourier_performance_k8_bandlimited.npz'
    file = np.load(filename, allow_pickle=True)

    errors_all = file['results']
    params = file['params'].item()
    imaging_schemes = params['imaging_schemes']
    fourier_coeffs = file['fourier_coeffs']
    peak_factors_size = errors_all.shape[0]
    sigmas_size = errors_all.shape[1]

    fig = plt.figure(figsize=(15, 3*peak_factors_size))
    gs = gridspec.GridSpec(peak_factors_size, sigmas_size, figure=fig, hspace=0.05, wspace=0.12)

    for i in range(peak_factors_size):
        for j in range(sigmas_size):
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i, j],
                                                        width_ratios=[1, 1], hspace=0, wspace=0.05)

            ax_low = fig.add_subplot(inner_gs[0, 0])
            ax_high = fig.add_subplot(inner_gs[0, 1], sharey=ax_low)

            if j == 0:
                ax_low.set_ylabel('RMSE (cm)')
            if i == peak_factors_size-1:
                ax_high.set_xlabel('# of Fourier Coeff.')
                ax_low.set_xlabel('# of Fourier Coeff.')



            ax_low.grid(True)
            ax_high.grid(True)

            plt.setp(ax_high.get_yticklabels(), visible=False)
            ax_high.tick_params(left=False)

            if i != peak_factors_size-1:
                ax_low.set_xticks(np.arange(fourier_coeffs.shape[0]))
                ax_low.set_xticklabels([
                    str(fourier_coeffs[i]) if i % 2 == 0 else ''
                    for i in range(fourier_coeffs.shape[0])
                ])
                ax_high.set_xticks(np.arange(fourier_coeffs.shape[0]))
                ax_high.set_xticklabels([
                    str(fourier_coeffs[i]) if i % 2 == 0 else ''
                    for i in range(fourier_coeffs.shape[0])
                ])
                plt.setp(ax_high.get_xticklabels(), visible=False)
                plt.setp(ax_low.get_xticklabels(), visible=False)
            else:
                ax_low.set_xticks(np.arange(fourier_coeffs.shape[0]))
                ax_low.set_xticklabels([
                    str(fourier_coeffs[i]) if i % 2 == 0 else ''
                    for i in range(fourier_coeffs.shape[0])
                ])
                ax_high.set_xticks(np.arange(fourier_coeffs.shape[0]))
                ax_high.set_xticklabels([
                    str(fourier_coeffs[i]) if i % 2 == 0 else ''
                    for i in range(fourier_coeffs.shape[0])
                ])

            for l in range(2):

                if imaging_schemes[l].coding_id.startswith('TruncatedFourier'):
                    str_name = 'Trunc. Fourier'
                elif imaging_schemes[l].coding_id == 'Identity':
                    str_name = 'FRH'
                elif imaging_schemes[l].coding_id == 'Greys':
                    str_name = 'Count. Gray'
                elif imaging_schemes[l].coding_id.startswith('Learned'):
                    str_name = 'Optimized'


                ax_low.plot(errors_all[i, j, :, l, 0] / 10, marker='o', linestyle='-', label=str_name,
                               color=get_scheme_color(imaging_schemes[l].coding_id, 8, cw_tof=imaging_schemes[l].cw_tof,
                                                          constant_pulse_energy=imaging_schemes[l].constant_pulse_energy))

                ax_high.plot(errors_all[i, j, :, l, 1] / 10, marker='o', linestyle='-', label=str_name,
                               color=get_scheme_color(imaging_schemes[l].coding_id, 8, cw_tof=imaging_schemes[l].cw_tof,
                                                          constant_pulse_energy=imaging_schemes[l].constant_pulse_energy))
            if i == 0:
                ax_low.set_title('Low SBR')
                ax_high.set_title('High SBR')
                ax_low.legend(fontsize=10)
                ax_high.legend(fontsize=10)


    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(f'fourier_peformance_grid_peak.svg', bbox_inches='tight')
    plt.show(block=True)


print()
print('YAYYY')
