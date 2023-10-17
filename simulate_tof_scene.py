# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace
mpl.use('qt5agg')
from indirect_toflib import indirect_tof_utils, CodingFunctions
from combined_toflib import combined_tof_utils
from direct_toflib import tirf
from direct_toflib import direct_tof_utils
from direct_toflib.direct_tof_utils import time2depth
import math
from PlotUtils import plot_signals
from toflib.coding import IdentityCoding

plot = 0
def combined_and_indirect_mae(params, depths, sbr, n_photons):

    n_tbins = params['n_tbins']
    tau = params['tau']
    K = params['K']
    meanBeta = params['meanBeta']
    dMax = params['dMax']
    trials = params['trials']

    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)

    Incident = indirect_tof_utils.GetIncident(ModFs, n_photons, meanBeta, sbr)

    Measures = indirect_tof_utils.GetMeasurements(ModFs, DemodFs)

    NormMeasures = (Measures.transpose() - np.mean(Measures, axis=1)) / np.std(Measures, axis=1)
    NormMeasures = NormMeasures.transpose()

    gt_depths = depths
    depths = np.round((depths / dMax) * n_tbins)
    ###DEPTH ESTIMATIONS
    measures_idtof = combined_tof_utils.IDTOF(Incident, DemodFs, depths, trials)
    measures_itof = indirect_tof_utils.ITOF(Incident, DemodFs, depths, trials)

    norm_measurements_idtof = indirect_tof_utils.NormalizeMeasureVals(measures_idtof, axis=1)
    norm_measurements_itof = indirect_tof_utils.NormalizeMeasureVals(measures_itof, axis=1)

    decoded_depths_idtof = np.argmax(np.dot(NormMeasures, norm_measurements_idtof.transpose()), axis=0)
    decoded_depths_itof = np.argmax(np.dot(NormMeasures, norm_measurements_itof.transpose()), axis=0)

    decoded_depths_itof = decoded_depths_itof * dMax / n_tbins
    decoded_depths_idtof = decoded_depths_idtof * dMax / n_tbins

    results={}
    results['mae_idtof'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_idtof)
    results['mae_itof'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_itof)

    return results


def direct_mae(params, depths, sbr, n_photons):

    n_tbins = params['n_tbins']
    tau = params['rep_tau']
    K = params['K']
    meanBeta = params['meanBeta']
    trials = params['trials']
    addPeak = params['addPeak']
    pw_factors = params['pw_factors']
    rec_algo = params['rec_algo']
    gt_depths = depths

    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)
    Incident = indirect_tof_utils.GetIncident(ModFs, n_photons, meanBeta, sbr=None)

    if (addPeak):peak_power = np.max(Incident)
    else: peak_power = None


    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (direct_tof_utils.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))
    gt_tshifts = direct_tof_utils.depth2time(depths)

    # Create GT gaussian pulses for each coding. Different coding may use different pulse widths
    pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors * tbin_res, mu=gt_tshifts, t_domain=t_domain)
    pulses = pulses_list[0]

    clean_pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors * tbin_res, mu=0, t_domain=t_domain)
    clean_pulses = clean_pulses_list[0]

    pulses.set_sbr(sbr)
    clean_pulses.set_sbr(None)

    simulated_pulses = pulses.simulate_n_signal_photons(n_photons=n_photons, n_mc_samples=trials, peak_power=peak_power, num_mes=K)
    clean_sim_pulses = clean_pulses.simulate_n_signal_photons(n_photons=n_photons, n_mc_samples=1, peak_power=peak_power, num_mes=K, add_noise=False)


    coding_obj = IdentityCoding(n_maxres=n_tbins, account_irf=False, h_irf=None)
    c_vals = coding_obj.encode(simulated_pulses)
    decoded_depths_dtof_maxguass = coding_obj.maxgauss_peak_decoding(c_vals, gauss_sigma=pw_factors[0],rec_algo_id=rec_algo) * tbin_depth_res
    decoded_depths_dtof_argmax = coding_obj.max_peak_decoding(c_vals, rec_algo_id=rec_algo) * tbin_depth_res


    Measures = combined_tof_utils.GetPulseMeasurements(clean_sim_pulses, DemodFs)
    NormMeasures = (Measures.transpose() - np.mean(Measures, axis=1)) / np.std(Measures, axis=1)
    NormMeasures = NormMeasures.transpose()


    measurements_pulsed_idtof = combined_tof_utils.pulses_idtof(simulated_pulses,DemodFs, trials)
    norm_measurements_idtof = indirect_tof_utils.NormalizeMeasureVals(measurements_pulsed_idtof, axis=1)
    decoded_depths_pulsed_idtof = np.argmax(np.dot(NormMeasures, norm_measurements_idtof.transpose()), axis=0)
    decoded_depths_pulsed_idtof = decoded_depths_pulsed_idtof * dMax / n_tbins

    results = {}
    results['mae_dtof_maxgauss'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_dtof_maxguass)
    results['mae_dtof_argmax'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_dtof_argmax)
    results['mae_pulsed_idtof'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_pulsed_idtof)

    if (plot):plot_signals(simulated_pulses, indirect_tof_utils.GetIncident(ModFs, n_photons, meanBeta, sbr))
    return results
