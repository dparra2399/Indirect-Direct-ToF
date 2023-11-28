# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
from IPython.core import debugger
breakpoint = debugger.set_trace
mpl.use('qt5agg')
from indirect_toflib import indirect_tof_utils, CodingFunctions
from combined_toflib import combined_tof_utils
from direct_toflib import tirf
from direct_toflib import direct_tof_utils
from research_utils import shared_constants, np_utils
import debug_utils
from indirect_toflib.coding import IdentityCoding, KTapSinusoidCoding

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
    T = params['T']

    tbin_depth_res = direct_tof_utils.time2depth(params['rep_tau'] / n_tbins)
    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)

    ModFs = indirect_tof_utils.ScaleMod(ModFs, tau=tau, pAveSource=pAveSource)
    Incident = indirect_tof_utils.GetIncident(ModFs, pAveSource, T=T, meanBeta=meanBeta, sbr=sbr,
                                              pAveAmbient=pAveAmbient, dt=dt, tau=tau)

    source_incident = indirect_tof_utils.GetIncident(ModFs, T=T, meanBeta=meanBeta)

    Measures = indirect_tof_utils.GetMeasurements(ModFs, DemodFs)

    NormMeasures = (Measures.transpose() - np.mean(Measures, axis=1)) / np.std(Measures, axis=1)
    NormMeasures = NormMeasures.transpose()

    gt_depths = depths
    depths = np.round((depths / dMax) * n_tbins)
    ###DEPTH ESTIMATIONS
    measures_idtof = combined_tof_utils.IDTOF(Incident, DemodFs, depths, trials, tbin_depth_res=tbin_depth_res, src_incident=source_incident)
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
    rec_algos = params['rec_algos']
    depth_res = params['depth_res']
    coding_schemes = params['coding_schemes']
    n_coding_schemes = len(coding_schemes)
    dMax = params['dMax']
    dt = params['dt']
    T = params['T']
    freq_idx = [1]
    gt_depths = depths
    depths = np.round((gt_depths / dMax) * n_tbins)

    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)
    ModFs = indirect_tof_utils.ScaleMod(ModFs, tau=tau, pAveSource=pAveSource)
    peak_power = np.max(ModFs)

    if (len(rec_algos) == 1): rec_algos = [rec_algos[0]] * n_coding_schemes
    # If only one pulse width is given, use that same pulse width for all coding
    if (len(pw_factors) == 1): pw_factors = np_utils.to_nparray([pw_factors[0]] * n_coding_schemes)

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

    coding_list = []
    ind_coding = IdentityCoding(n_maxres=n_tbins, account_irf=False, h_irf=None)
    coding_list.append(ind_coding)
    sin_coding = KTapSinusoidCoding(n_maxres=n_tbins, freq_idx=freq_idx, k=K,account_irf=False, h_irf=None)
    coding_list.append(sin_coding)

    results = {}


    for k in range(n_coding_schemes):
        coding_obj = coding_list[k]
        rec_algo = rec_algos[k]
        pulses_pp = pulses_list_pp[k] #Peak power pulse
        pulses_ave = pulses_list_ave[k] #Ave Power pulse

        #Set pulse sbr
        pulses_pp.set_sbr(sbr)
        pulses_ave.set_sbr(sbr)
        #Set pulse ambient light
        pulses_pp.set_ambient(pAveAmbient)
        pulses_ave.set_ambient(pAveAmbient)

        #Set pulse reflivity
        pulses_pp.set_mean_beta(meanBeta)
        pulses_ave.set_mean_beta(meanBeta)
        #Set Integration time
        pulses_pp.set_integration_time(T)
        pulses_ave.set_integration_time(T)

        #PEAK POWER PULSES
        num_m = K
        if coding_schemes[k] == 'KTapSinusoid':
            num_m = 1
        simulated_pulses_pp = pulses_pp.simulate_peak_power(peak_power, pAveSource=pAveSource,
                                                            dt=dt, tau=tau, num_measures=num_m, n_mc_samples=trials)
        #AVERAGE POWER PULSES
        simulated_pulses_ave = pulses_ave.simulate_avg_power(pAveSource, n_mc_samples=trials, dt=dt, tau=tau)

        c_vals_pp = coding_obj.encode(simulated_pulses_pp)
        c_vals_ave = coding_obj.encode(simulated_pulses_ave)

        if coding_schemes[k] == 'Identity':
            decoded_depths_pp = coding_obj.maxgauss_peak_decoding(c_vals_pp, gauss_sigma=pw_factors[0],
                                                                                rec_algo_id=rec_algo) * tbin_depth_res
            decoded_depths_ave = coding_obj.maxgauss_peak_decoding(c_vals_ave, gauss_sigma=pw_factors[0],
                                                                                 rec_algo_id=rec_algo) * tbin_depth_res
        else:
            decoded_depths_pp = coding_obj.max_peak_decoding(c_vals_pp, rec_algo_id=rec_algo) * tbin_depth_res
            decoded_depths_ave = coding_obj.max_peak_decoding(c_vals_ave, rec_algo_id=rec_algo) * tbin_depth_res

        results[coding_schemes[k] + '_PP'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_pp) * depth_res
        results[coding_schemes[k] + '_AVE'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_ave) * depth_res


    if (shared_constants.debug):
        #debug_utils.debug_direct(params, c_vals_pp, pulses_pp, decoded_depths_dtof_maxguass_pp,
        #         decoded_depths_dtof_maxguass_ave, c_vals_ave, pulses_ave, depths, t_domain, sigma,
        #         peak_power, K, sbr, pAveAmbient, pAveSource)
        pass
    return results



