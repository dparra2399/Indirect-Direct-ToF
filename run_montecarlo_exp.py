# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace
mpl.use('qt5agg')
from simulate_tof_scene import combined_and_indirect_mae, direct_mae


def run_experiment(params, depths, sbr_levels, pAveAmbient_levels, pAveSource_levels):

    (n_noise_lvls,n_photon_lvls) = pAveSource_levels.shape

    results = {}
    results['mae_idtof'] = np.zeros(pAveSource_levels.shape)
    results['mae_itof'] = np.zeros(pAveSource_levels.shape)
    results['mae_dtof_argmax'] = np.zeros(pAveSource_levels.shape)
    results['mae_dtof_maxgauss'] = np.zeros(pAveSource_levels.shape)
    results['mae_pulsed_idtof'] = np.zeros(pAveSource_levels.shape)


    for x in range(0, n_noise_lvls):
        for y in range(0, n_photon_lvls):
            if sbr_levels is not None:
                sbr = sbr_levels[x, y]
            else:
                sbr = None

            if pAveSource_levels is not None:
                pAveAmbient = pAveAmbient_levels[x, y]
            else:
                pAveAmbient = None

            pAveSource = pAveSource_levels[x, y]

            results_indirect = combined_and_indirect_mae(params, depths, sbr, pAveAmbient, pAveSource)

            results_direct = direct_mae(params, depths, sbr, pAveAmbient, pAveSource)

            results['mae_idtof'][x, y] = results_indirect['mae_idtof']
            results['mae_itof'][x, y] = results_indirect['mae_itof']
            results['mae_dtof_argmax'][x, y] = results_direct['mae_dtof_argmax']
            results['mae_dtof_maxgauss'][x,y] = results_direct['mae_dtof_maxgauss']
            results['mae_pulsed_idtof'][x, y] = results_direct['mae_pulsed_idtof']

    return results



