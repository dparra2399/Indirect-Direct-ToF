# Python imports
# Library imports
import numpy as np
from IPython.core import debugger
import time

import simulate_tof_scene
from run_montecarlo_exp import run_experiment, run_both
from direct_toflib import direct_tof_utils
from research_utils import shared_constants
import matplotlib.pyplot as plt

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
    params['depth_res'] = 1000 ##Conver to MM
    params['dt'] = params['rep_tau'] / float(params['n_tbins'])
    params['coding_functions'] = ['KTapSinusoid', 'HamiltonianK4']
    params['meanBeta'] = 1e-4
    # Avg fraction of photons reflected from a scene points back to the detector

    ##DIRECT
    params['pw_factors'] = np.array([30])
    params['peak_factor'] = 2
    params['freq_idx'] = [1]
    #params['time_res'] = 0.5 * 1e-9 #1 nanosecond
    #params['time_res'] = 50 * 1e-12 #50 picoseconds
    params['time_res'] = None
    params['rec_algos'] = ['matchfilt', 'zncc', 'zncc']
    params['coding_schemes'] = ['Identity', 'KTapSinusoid', 'HamiltonianK4']
    params['trials'] = 1

    depths = np.array([11.4])

    run_exp = 0
    exp_num = 0

    n_signal_lvls = 20
    n_sbr_lvls = 20

    (min_power_exp, max_power_exp) = (4, 8)
    (min_sbr_exp, max_sbr_exp) = (-1, 1)
    (min_amb_exp, max_amb_exp) = (2, 8)

    pAveSource_levels_list = np.round(np.power(10, np.linspace(min_power_exp, max_power_exp, n_signal_lvls)))
    pAveAmbient_levels_list = np.round(np.power(10, np.linspace(min_amb_exp, max_amb_exp, n_signal_lvls)))
    sbr_levels_list = np.power(10, np.linspace(min_sbr_exp, max_sbr_exp, n_sbr_lvls))
    sbr_levels, pAveSource_levels = np.meshgrid(sbr_levels_list, pAveSource_levels_list)
    pAveAmbient_levels, _ = np.meshgrid(pAveAmbient_levels_list, pAveSource_levels_list)

    pAveSource = (10**4)
    # pAveAmbient = (10**5)
    pAveAmbient = None
    sbr = 10**-1

    if params['time_res'] is not None:
        params['n_tbins'] = int(np.round((params['rep_tau'] * np.squeeze(params['pw_factors'])) / params['time_res']))
        print("Optimized number of time bins:, ", params['n_tbins'])


    if run_exp:
        tic = time.perf_counter()
        results = run_experiment(params, depths, sbr_levels, None, pAveSource_levels)

        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")

        FileUtils.WriteErrorsToFile(
            params=params, results=results, exp_num=exp_num, depths=depths,
            sbr_levels=sbr_levels, pAveAmbient_levels=None, pAveSource_levels=pAveSource_levels)


    else:
        tic = time.perf_counter()
        (results_indirect, results_direct) = run_both(params, depths, sbr, pAveAmbient, pAveSource)

        toc = time.perf_counter()
        coding_functions = params['coding_functions']
        for k in range(len(coding_functions)):
            print()
            print(f"MAE {coding_functions[k]} IDTOF: {results_indirect[coding_functions[k] + '_IDTOF']: .3f},")
            print(f"MAE {coding_functions[k]} ITOF: {results_indirect[coding_functions[k] + '_ITOF']: .3f},")

        coding_schemes = params['coding_schemes']
        for k in range(len(coding_schemes)):
            print()
            print(f"MAE DTOF {coding_schemes[k]} Peak Power: {results_direct[coding_schemes[k] + '_PP']: .3f}")
            print(f"MAE DTOF {coding_schemes[k]} Average Power: {results_direct[coding_schemes[k] + '_AVE']: .3f}")

        print(f"Completed in {toc - tic:0.4f} seconds")


if (shared_constants.debug):
    plt.show()
print('Complete')

