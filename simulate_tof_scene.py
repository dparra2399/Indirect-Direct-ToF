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
from research_utils import shared_constants
import debug_utils
import math
from toflib.coding import IdentityCoding

plot = 0
def combined_and_indirect_mae(params, depths, sbr, pAveAmbient, pAveSource):

    n_tbins = params['n_tbins']
    tau = params['tau']
    K = params['K']
    meanBeta = params['meanBeta']
    dMax = params['dMax']
    dt = params['dt']
    trials = params['trials']
    peak_factor = params['peak_factor']
    depth_res = params['depth_res']
    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)

    ModFs = indirect_tof_utils.ScaleMod(ModFs, tau=tau, pAveSource=pAveSource)
    Incident = indirect_tof_utils.GetIncident(ModFs, pAveSource, meanBeta=meanBeta, sbr=sbr,
                                              pAveAmbient=pAveAmbient, dt=dt, tau=tau)

    Measures = indirect_tof_utils.GetMeasurements(ModFs, DemodFs)

    NormMeasures = (Measures.transpose() - np.mean(Measures, axis=1)) / np.std(Measures, axis=1)
    NormMeasures = NormMeasures.transpose()

    gt_depths = depths
    depths = np.round((depths / dMax) * n_tbins)
    ###DEPTH ESTIMATIONS
    measures_idtof = combined_tof_utils.IDTOF(Incident, DemodFs, depths, trials)
    measures_itof = indirect_tof_utils.ITOF(Incident, DemodFs, depths, trials)

    if (shared_constants.debug):
        new_measures_idtof = measures_idtof[depths.astype(int), :]
        new_measures_itof = measures_itof[depths.astype(int), :]
    else:
        new_measures_idtof = measures_idtof
        new_measures_itof = measures_itof

    norm_measurements_idtof = indirect_tof_utils.NormalizeMeasureVals(new_measures_idtof, axis=1)
    norm_measurements_itof = indirect_tof_utils.NormalizeMeasureVals(new_measures_itof, axis=1)

    decoded_depths_idtof = np.argmax(np.dot(NormMeasures, norm_measurements_idtof.transpose()), axis=0)
    decoded_depths_itof = np.argmax(np.dot(NormMeasures, norm_measurements_itof.transpose()), axis=0)

    decoded_depths_itof = decoded_depths_itof * dMax / n_tbins
    decoded_depths_idtof = decoded_depths_idtof * dMax / n_tbins

    results={}
    results['mae_idtof'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_idtof) * depth_res
    results['mae_itof'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_itof) * depth_res

    if (shared_constants.debug):
        debug_utils.debbug_cw(params, ModFs, Incident, NormMeasures, norm_measurements_idtof, norm_measurements_itof,
                  depths, sbr, pAveAmbient, pAveSource)

    return results


def direct_mae(params, depths, sbr, pAveAmbient, pAveSource):

    n_tbins = params['n_tbins']
    tau = params['rep_tau']
    K = params['K']
    meanBeta = params['meanBeta']
    trials = params['trials']
    pw_factors = params['pw_factors']
    peak_factor = params['peak_factor']
    rec_algo = params['rec_algo']
    depth_res = params['depth_res']
    dMax = params['dMax']
    dt = params['dt']
    gt_depths = depths
    depths = np.round((gt_depths / dMax) * n_tbins)

    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)
    ModFs = indirect_tof_utils.ScaleMod(ModFs, tau=tau, pAveSource=pAveSource)
    peak_power = np.max(ModFs)

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (direct_tof_utils.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))

    #tbin_res is size of bin in time
    #tbin_depth is size of bin in depth
    #t_domain is time associated with each bin
    #rep_freq is number of pulses per second
    #rep_tau is max time
    #dMax is max depth

    gt_tshifts = direct_tof_utils.depth2time(gt_depths)

    # Create GT gaussian pulses for each coding. Different coding may use different pulse widths
    sigma = pw_factors * tbin_res
    pulses_list_pp = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=gt_tshifts, t_domain=t_domain)
    pulses_list_ave = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=gt_tshifts, t_domain=t_domain)
    pulses_pp = pulses_list_pp[0] #Peak power pulse
    pulses_ave = pulses_list_ave[0] #Ave Power pulse

    all_depths = np.linspace(0, params['dMax'], n_tbins)
    all_tshifts = direct_tof_utils.depth2time(all_depths)
    clean_pulse_list_pp = tirf.init_gauss_pulse_list(n_tbins, pw_factors * tbin_res, mu=all_tshifts, t_domain=t_domain)
    clean_pulse_list_ave = tirf.init_gauss_pulse_list(n_tbins, pw_factors * tbin_res, mu=all_tshifts, t_domain=t_domain)
    clean_pulse_pp = clean_pulse_list_pp[0]
    clean_pulse_ave = clean_pulse_list_ave[0]

    #Set pulse sbr
    pulses_pp.set_sbr(sbr)
    pulses_ave.set_sbr(sbr)
    clean_pulse_pp.set_sbr(None)
    clean_pulse_ave.set_sbr(None)
    #Set pulse ambient light
    pulses_pp.set_ambient(pAveAmbient)
    pulses_ave.set_ambient(pAveAmbient)
    clean_pulse_pp.set_ambient(None)
    clean_pulse_ave.set_ambient(None)
    #Set pulse reflivity
    pulses_pp.set_mean_beta(meanBeta)
    pulses_ave.set_mean_beta(meanBeta)
    clean_pulse_pp.set_mean_beta(meanBeta)
    clean_pulse_ave.set_mean_beta(meanBeta)

    #PEAK POWER PULSES
    simulated_pulses_pp = pulses_pp.simulate_peak_power(peak_power, pAveSource=pAveSource,
                                                        dt=dt, tau=tau, num_measures=K, n_mc_samples=trials)
    clean_sim_pulses_pp = clean_pulse_pp.simulate_peak_power(peak_power, pAveSource=pAveSource,
                                                            dt=dt, tau=tau, num_measures=K, n_mc_samples=trials, add_noise=False)
    #AVERAGE POWER PULSES
    simulated_pulses_ave = pulses_ave.simulate_avg_power(pAveSource, n_mc_samples=trials, dt=dt, tau=tau)
    clean_sim_pulses_ave = clean_pulse_ave.simulate_avg_power(pAveSource, n_mc_samples=trials,
                                                              dt=dt, tau=tau, add_noise=False)
    #COMBINED CASE WITH PULSES
    Incident_pulses_pp = indirect_tof_utils.GetIncident(clean_sim_pulses_pp, pAveSource, meanBeta=meanBeta,
                                                        sbr=sbr, pAveAmbient=pAveAmbient, dt=dt, tau=tau)
    Incident_pulses_ave = indirect_tof_utils.GetIncident(clean_sim_pulses_ave, pAveSource, meanBeta=meanBeta,
                                                        sbr=sbr, pAveAmbient=pAveAmbient, dt=dt, tau=tau)


    coding_obj = IdentityCoding(n_maxres=n_tbins, account_irf=False, h_irf=None)
    c_vals_pp = coding_obj.encode(simulated_pulses_pp)
    c_vals_ave = coding_obj.encode(simulated_pulses_ave)

    #DECODE PEAK POWER DEPTHS
    decoded_depths_dtof_maxguass_pp = coding_obj.maxgauss_peak_decoding(c_vals_pp, gauss_sigma=pw_factors[0],rec_algo_id=rec_algo) * tbin_depth_res
    decoded_depths_dtof_argmax_pp = coding_obj.max_peak_decoding(c_vals_pp, rec_algo_id=rec_algo) * tbin_depth_res

    #DECODE AVERAGE POWER DEPTHS
    decoded_depths_dtof_maxguass_ave = coding_obj.maxgauss_peak_decoding(c_vals_ave, gauss_sigma=pw_factors[0], rec_algo_id=rec_algo) * tbin_depth_res
    decoded_depths_dtof_argmax_ave = coding_obj.max_peak_decoding(c_vals_ave, rec_algo_id=rec_algo) * tbin_depth_res

    #CLEAN MEASURES FOR PEAK POWER PULSES
    Measures_pp = combined_tof_utils.GetPulseMeasurements(clean_sim_pulses_pp, DemodFs)
    NormMeasures_pp = (Measures_pp.transpose() - np.mean(Measures_pp, axis=1)) / np.std(Measures_pp, axis=1)
    NormMeasures_pp = NormMeasures_pp.transpose()

    #CLEAN MEASURES FOR AVERGAE POWER PULSES
    Measures_ave = combined_tof_utils.GetPulseMeasurements(clean_sim_pulses_ave, DemodFs)
    NormMeasures_ave = (Measures_ave.transpose() - np.mean(Measures_ave, axis=1)) / np.std(Measures_ave, axis=1)
    NormMeasures_ave = NormMeasures_ave.transpose()

    #ACTUAL MEASURES FOR PEAK POWER AND AVERAGE POWER
    measurements_pulsed_idtof_pp = combined_tof_utils.pulses_idtof(Incident_pulses_pp,DemodFs, depths, trials)
    norm_measurements_idtof_pp = indirect_tof_utils.NormalizeMeasureVals(measurements_pulsed_idtof_pp, axis=1)

    measurements_pulsed_idtof_ave = combined_tof_utils.pulses_idtof(Incident_pulses_ave, DemodFs, depths, trials)
    norm_measurements_idtof_ave = indirect_tof_utils.NormalizeMeasureVals(measurements_pulsed_idtof_ave, axis=1)

    #DECODED DEAPTHS FOR COMBINED CASE
    decoded_depths_pulsed_idtof_pp = np.argmax(np.dot(NormMeasures_pp, norm_measurements_idtof_pp.transpose()), axis=0)
    decoded_depths_pulsed_idtof_pp = decoded_depths_pulsed_idtof_pp * dMax / n_tbins

    decoded_depths_pulsed_idtof_ave = np.argmax(np.dot(NormMeasures_ave, norm_measurements_idtof_ave.transpose()), axis=0)
    decoded_depths_pulsed_idtof_ave = decoded_depths_pulsed_idtof_ave * dMax / n_tbins

    results = {}
    results['mae_dtof_maxgauss_pp'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_dtof_maxguass_pp) * depth_res
    results['mae_dtof_argmax_pp'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_dtof_argmax_pp) * depth_res
    results['mae_pulsed_idtof_pp'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_pulsed_idtof_pp) * depth_res

    results['mae_dtof_maxgauss_ave'] = indirect_tof_utils.ComputeMetrics(gt_depths,decoded_depths_dtof_maxguass_ave) * depth_res
    results['mae_dtof_argmax_ave'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_dtof_argmax_ave) * depth_res
    results['mae_pulsed_idtof_ave'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_pulsed_idtof_ave) * depth_res

    if (shared_constants.debug):
        debug_utils.debug_direct(params, c_vals_pp, pulses_pp, decoded_depths_dtof_maxguass_pp,
                 decoded_depths_dtof_maxguass_ave, c_vals_ave, pulses_ave, depths, t_domain, sigma,
                 peak_power, K, sbr, pAveAmbient, pAveSource)

    return results



