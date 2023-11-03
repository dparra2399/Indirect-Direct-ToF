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
    params['n_tbins'] = 1024
    params['K'] = 4
    params['T'] = 0.1  # Integration time. Exposure time in seconds
    params['rep_freq'] = 1e+7
    params['fund_freq'] = params['rep_freq']
    params['rep_tau'] = 1. / params['rep_freq']
    params['tau'] = 1./ params['fund_freq']
    params['dMax'] = direct_tof_utils.time2depth(params['rep_tau'])
    params['depth_res'] = 1000
    params['dt'] = params['rep_tau'] / float(params['n_tbins'])
    params['meanBeta'] = 1e-4
    # Avg fraction of photons reflected from a scene points back to the detector

    ##DIRECT
    params['pw_factors'] = np.array([1])
    params['peak_factor'] = 1
    params['rec_algo'] = 'linear'
    params['trials'] = 3000

    depths = np.array([3.42, 8.78, 12.22, 13.68, 1.34])

    run_exp = 1
    exp_num = 'tmp'

    n_signal_lvls = 20
    n_sbr_lvls = 20

    (min_power_exp, max_power_exp) = (4, 10)
    (min_sbr_exp, max_sbr_exp) = (-1, 1)
    (min_amb_exp, max_amb_exp) = (2, 8)

    pAveSource_levels_list = np.round(np.power(10, np.linspace(min_power_exp, max_power_exp, n_signal_lvls)))
    pAveAmbient_levels_list = np.round(np.power(10, np.linspace(min_amb_exp, max_amb_exp, n_signal_lvls)))
    sbr_levels_list = np.power(10, np.linspace(min_sbr_exp, max_sbr_exp, n_sbr_lvls))
    sbr_levels, pAveSource_levels = np.meshgrid(sbr_levels_list, pAveSource_levels_list)
    pAveAmbient_levels, _ = np.meshgrid(pAveAmbient_levels_list, pAveSource_levels_list)

    pAveSource = (10**4)
    pAveAmbient = (10**8)
    #pAveAmbient = None
    sbr = None

    if run_exp:
        tic = time.perf_counter()
        results = run_experiment(params, depths, None, pAveAmbient_levels, pAveSource_levels)

        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")

        FileUtils.WriteErrorsToFile(
            params=params, results=results, exp_num=exp_num, depths=depths,
            sbr_levels=sbr_levels, pAveAmbient_levels= pAveAmbient_levels, pAveSource_levels=pAveSource_levels)


    else:
        tic = time.perf_counter()
        results_indirect = simulate_tof_scene.combined_and_indirect_mae(params, depths, sbr, pAveAmbient, pAveSource)

        results_direct = simulate_tof_scene.direct_mae(params, depths, sbr, pAveAmbient, pAveSource)

        toc = time.perf_counter()

        print(f"MAE IDTOF: {results_indirect['mae_idtof']: .3f},")
        print(f"MAE ITOF: {results_indirect['mae_itof']: .3f},")
        print(f"MAE DTOF (Argmax): {results_direct['mae_dtof_argmax']: .3f}")
        print(f"MAE DTOF (Maxgauss): {results_direct['mae_dtof_maxgauss']: .3f}")
        print(f"MAE Pulsed IDTOF: {results_direct['mae_pulsed_idtof']: .3f},")

        print(f"Completed in {toc - tic:0.4f} seconds")

print('hello world')