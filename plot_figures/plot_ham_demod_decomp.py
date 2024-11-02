from IPython.core import debugger

from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from spad_toflib.spad_tof_utils import split_into_indices, gated_ham

from plot_figures.plot_utils import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Qt5Agg')

font = {'family': 'serif',
            'size': 7}

matplotlib.rc('font', **font)

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
            ImagingSystemParams('HamiltonianK3', 'HamiltonianK3', 'zncc',
                                duty=1. / 6., freq_window=0.1),
            ImagingSystemParams('HamiltonianK4', 'HamiltonianK4', 'zncc',
                                duty=1. / 6., freq_window=0.1),
            ImagingSystemParams('HamiltonianK5', 'HamiltonianK5', 'zncc',
                                duty=1. / 6., freq_window=0.1)
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

        fig, axs = plt.subplots(len(imaging_schemes), 5, figsize=(10, 8))

        for k in range(len(imaging_schemes)):
            imaging_scheme = imaging_schemes[k]
            coding_obj = imaging_scheme.coding_obj
            coding_scheme = imaging_scheme.coding_id
            light_obj = imaging_scheme.light_obj
            light_source = imaging_scheme.light_id
            rec_algo = imaging_scheme.rec_algo

            demodfs = coding_obj.demodfs

            total_gates = np.zeros((n_tbins, 1))
            for i in range(coding_obj.n_functions):
                gates = gated_ham(demodfs[:, i])
                total_gates = np.hstack((total_gates, gates))
                indices = split_into_indices(demodfs[:, i])
                for j in range(len(indices)-1):
                    idx1 = indices[j][1]
                    idx2 = indices[j+1][0]
                    middle = int((idx1 + idx2) // 2)
                    axs[k][i].axvline(middle, color='black', linestyle='--')
                axs[k][i].set_xticks([])
                axs[k][i].set_yticks([])
                axs[k][i].spines['top'].set_visible(False)
                axs[k][i].spines['right'].set_visible(False)
                axs[k][i].set_title(r'$d_{' + str(i) + r'}(t)$')
                axs[k][i].set_xlabel('Time')

                axs[k][i].plot(gates)

            for p in range(i+1, 5):
                axs[k][p].set_axis_off()

        fig.tight_layout()
        fig.savefig(os.path.join(save_folder, 'suppfigure3.svg'), bbox_inches='tight')

        plt.show()