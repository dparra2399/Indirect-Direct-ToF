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
    #irf=None
    #irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 30, circ_shifted=True)
    irf = np.load(r"C:\Users\Patron\PycharmProjects\WISC-SinglePhoton3DData\system_irf\20190207_face_scanning_low_mu\free\irf_tres-8ps_tlen-17504ps.npy")
    params['imaging_schemes'] = [
        ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, h_irf=irf),

    ]


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
    peak_factor = 0.05


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

    imaging_scheme = imaging_schemes[0]
    coding_obj = imaging_scheme.coding_obj
    coding_scheme = imaging_scheme.coding_id
    light_obj = imaging_scheme.light_obj
    light_source = imaging_scheme.light_id
    rec_algo = imaging_scheme.rec_algo

    incident = np.squeeze(light_obj.simulate_average_photons(photon_count, sbr, peak_factor=peak_factor))

    filtered_illum = np.roll(incident[0, :], int(n_tbins // 2))
    illum = np.roll(light_obj.light_source, int(n_tbins // 2))

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(illum, color='blue')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].plot(filtered_illum, color='blue')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig('Z:\\Research_Users\\David\\Learned Coding Functions Paper\\teaser_illum_figure_part2.svg', bbox_inches='tight')
    plt.show(block=True)

print()
print('YAYYY')
