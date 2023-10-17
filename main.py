# Python imports
# Library imports
import numpy as np
from IPython.core import debugger
import time

import simulate_tof_scene
from run_montecarlo_exp import run_experiment
from direct_toflib import direct_tof_utils

breakpoint = debugger.set_trace

import FileUtils

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    params = {}
    params['n_tbins'] = 256
    params['K'] = 4
    params['T'] = 0.1  # Integration time. Exposure time in seconds
    speedOfLight = 299792458. * 1000.  # mm / sec
    params['rep_freq'] = 1e+7
    params['fund_freq'] = params['rep_freq']
    params['rep_tau'] = 1. / params['rep_freq']
    params['tau'] = 1./ params['fund_freq']
    params['dMax'] = direct_tof_utils.time2depth(params['rep_tau'])
    params['dt'] = params['rep_tau'] / float(params['n_tbins'])
    params['meanBeta'] = 1
    # Avg fraction of photons reflected from a scene points back to the detector

    ##DIRECT
    params['pw_factors'] = np.array([8])
    params['addPeak'] = True
    params['rec_algo'] = 'linear'
    params['trials'] = 1000

    depths = np.array([5.32, 6.78, 2.01, 7.68, 8.34])

    run_exp = 1
    exp_num = 0

    n_signal_lvls = 20
    n_sbr_lvls = 20

    (min_signal_exp, max_signal_exp) = (0.5, 4.5)
    (min_sbr_exp, max_sbr_exp) = (-1, 1)
    photon_levels_list = np.round(np.power(10, np.linspace(min_signal_exp, max_signal_exp, n_signal_lvls)))
    sbr_levels_list = np.power(10, np.linspace(min_sbr_exp, max_sbr_exp, n_sbr_lvls))
    sbr_levels, photon_levels = np.meshgrid(sbr_levels_list, photon_levels_list)

    n_photons = 10
    sbr = 0.5

    if run_exp:
        tic = time.perf_counter()
        results = run_experiment(params, depths, sbr_levels, photon_levels)

        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")

        FileUtils.WriteErrorsToFile(
            params=params, results=results, exp_num=exp_num, depths=depths,
            sbr_levels=sbr_levels, photon_levels=photon_levels)


    else:
        tic = time.perf_counter()
        results_indirect = simulate_tof_scene.combined_and_indirect_mae(params, depths, sbr, n_photons)

        results_direct = simulate_tof_scene.direct_mae(params, depths, sbr, n_photons)

        toc = time.perf_counter()

        print(f"MAE IDTOF: {results_indirect['mae_idtof']: .3f},")
        print(f"MAE ITOF: {results_indirect['mae_itof']: .3f},")
        print(f"MAE DTOF (Argmax): {results_direct['mae_dtof_argmax']: .3f}")
        print(f"MAE DTOF (Maxgauss): {results_direct['mae_dtof_maxgauss']: .3f}")
        print(f"MAE Pulsed IDTOF: {results_direct['mae_pulsed_idtof']: .3f},")

        print(f"Completed in {toc - tic:0.4f} seconds")

print('hello world')