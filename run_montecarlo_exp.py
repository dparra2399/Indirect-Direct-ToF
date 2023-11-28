# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace
mpl.use('qt5agg')
from simulate_tof_scene import combined_and_indirect_mae, direct_mae
from research_utils import shared_constants
import simulate_tof_scene


def run_experiment(params, depths, sbr_levels, pAveAmbient_levels, pAveSource_levels):

    (n_noise_lvls,n_photon_lvls) = pAveSource_levels.shape
    coding_schemes = params['coding_schemes']
    results = {}
    results['mae_idtof'] = np.zeros(pAveSource_levels.shape)
    results['mae_itof'] = np.zeros(pAveSource_levels.shape)

    for k in range(len(coding_schemes)):
        results[coding_schemes[k] + '_PP'] = np.zeros(pAveSource_levels.shape)
        results[coding_schemes[k] + '_AVE'] = np.zeros(pAveSource_levels.shape)

    assert sbr_levels is None or pAveAmbient_levels is None, "sbr or ambient lists must be none"
    for x in range(0, n_noise_lvls):
        for y in range(0, n_photon_lvls):
            if sbr_levels is not None:
                sbr = sbr_levels[x, y]
            else:
                sbr = None

            if pAveAmbient_levels is not None:
                pAveAmbient = pAveAmbient_levels[x, y]
            else:
                pAveAmbient = None

            pAveSource = pAveSource_levels[x, y]

            (results_indirect, results_direct) = run_both(params, depths, sbr, pAveAmbient, pAveSource)

            results['mae_idtof'][x, y] = results_indirect['mae_idtof']
            results['mae_itof'][x, y] = results_indirect['mae_itof']

            for k in range(len(coding_schemes)):
                results[coding_schemes[k] + '_PP'][x, y] = results_direct[coding_schemes[k] + '_PP']
                results[coding_schemes[k] + '_AVE'][x, y] = results_direct[coding_schemes[k] + '_AVE']


    return results

def run_both(params, depths, sbr, pAveAmbient, pAveSource):

    results_indirect = simulate_tof_scene.combined_and_indirect_mae(params, depths, sbr, pAveAmbient, pAveSource)
    results_direct = simulate_tof_scene.direct_mae(params, depths, sbr, pAveAmbient, pAveSource)


    return (results_indirect, results_direct)