# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
from IPython.core import debugger
breakpoint = debugger.set_trace
mpl.use('qt5agg')
import matplotlib.pyplot as plt
from indirect_toflib import indirect_tof_utils, CodingFunctions
from combined_toflib import combined_tof_utils
from direct_toflib import tirf
from direct_toflib import direct_tof_utils
from research_utils import shared_constants, np_utils
import debug_utils
from direct_toflib.coding import IdentityCoding, KTapSinusoidCoding
from direct_toflib.coding_utils import init_coding_list
from indirect_toflib.CodingFunctions_utils import init_coding_functions_list

plot = 0
def combined_and_indirect_mae(params, depths, sbr, pAveAmbient, pAveSource, coding_list=None):

    n_tbins = params['n_tbins']
    tau = params['tau']
    meanBeta = params['meanBeta']
    dMax = params['dMax']
    dt = params['dt']
    trials = params['trials']
    peak_factor = params['peak_factor']
    peak_power = peak_factor * pAveSource
    depth_res = params['depth_res']
    T = params['T']
    coding_functions = params['coding_functions']
    n_coding_functions = len(coding_functions)
    tbin_depth_res = direct_tof_utils.time2depth(params['rep_tau'] / n_tbins)

    if (coding_list is None): coding_list = init_coding_functions_list(coding_functions, n_tbins, params)
    results = {}
    gt_depths = depths
    depths = np.round((depths / dMax) * n_tbins)
    for i in range(n_coding_functions):
        (ModFs, DemodFs) = coding_list[i]

        ModFs = indirect_tof_utils.ScaleMod(ModFs, tau=tau, pAveSource=pAveSource)
        for k in range(0, ModFs.shape[-1]):
            ModFs[:, k] = ModFs[:, k] / params['num_measures'][i]

        Incident = indirect_tof_utils.GetIncident(ModFs, pAveSource, T=T, meanBeta=meanBeta, sbr=sbr,
                                                  pAveAmbient=pAveAmbient, dt=dt, tau=tau)

        Measures = indirect_tof_utils.GetMeasurements(ModFs, DemodFs)

        gated = False
        if coding_functions[i] in ['HamiltonianK3Gated', 'HamiltonianK4Gated', 'HamiltonianK5Gated']:
            gated = True

        NormMeasures = (Measures.transpose() - np.mean(Measures, axis=1)) / np.std(Measures, axis=1)
        NormMeasures = NormMeasures.transpose()

        ###DEPTH ESTIMATIONS
        measures_idtof = combined_tof_utils.IDTOF(Incident, DemodFs, depths, trials, gated=gated)
        measures_itof = indirect_tof_utils.ITOF(Incident, DemodFs, depths, trials, gated=gated)

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

        results[coding_functions[i] + '_IDTOF'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_idtof) * depth_res
        results[coding_functions[i] + '_ITOF'] = indirect_tof_utils.ComputeMetrics(gt_depths, decoded_depths_itof) * depth_res

    if (shared_constants.debug):
        plt.show()
    return results


def direct_mae(params, depths, sbr, pAveAmbient, pAveSource, coding_list=None, pulses_list_pp=None, pulses_list_ave=None):

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
    peak_factor = params['peak_factor']
    n_coding_schemes = len(coding_schemes)
    dMax = params['dMax']
    dt = params['dt']
    T = params['T']
    gt_depths = depths
    depths = np.round((gt_depths / dMax) * n_tbins)

    peak_power = pAveSource * peak_factor

    if (len(rec_algos) == 1): rec_algos = [rec_algos[0]] * n_coding_schemes
    # If only one pulse width is given, use that same pulse width for all coding
    if (len(pw_factors) == 1): pw_factors = np_utils.to_nparray([pw_factors[0]] * n_coding_schemes)

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (direct_tof_utils.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))

    #print('Max Depth:', dMax)
    #print('tbin_res:', tbin_res)
    #tbin_res is size of bin in time
    #tbin_depth is size of bin in depth
    #t_domain is time associated with each bin
    #rep_freq is number of pulses per second
    #rep_tau is max time
    #dMax is max depth

    gt_tshifts = direct_tof_utils.depth2time(gt_depths)

    # Create GT gaussian pulses for each coding. Different coding may use different pulse widths
    sigma = pw_factors * tbin_res
    if (pulses_list_pp is None): pulses_list_pp = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=gt_tshifts, t_domain=t_domain)
    if (pulses_list_ave is None): pulses_list_ave = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=gt_tshifts, t_domain=t_domain)

    if (coding_list is None): coding_list = init_coding_list(coding_schemes, n_tbins, params)
    results = {}

    if (shared_constants.debug):
        figure, axis = plt.subplots(n_coding_schemes)

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
        scheme = coding_schemes[k]
        num_m = 1
        noise = True
        if scheme in ['Identity']:
            num_m = K
        elif scheme in ['IntegratedGated', 'Gated']:
            num_m = 1 / params['n_gates']
            noise = False


        simulated_pulses_pp = pulses_pp.simulate_peak_power(peak_power, pAveSource=pAveSource,
                                                            dt=dt, tau=tau, num_measures=num_m, n_mc_samples=trials,
                                                            add_noise=noise)
        #AVERAGE POWER PULSES
        simulated_pulses_ave = pulses_ave.simulate_avg_power(pAveSource, num_measurements=num_m,
                                                             n_mc_samples=trials, dt=dt,tau=tau, add_noise=noise)

        if scheme == 'IntegratedGated':
            c_vals_pp = coding_obj.encode(simulated_pulses_pp, trials)
            c_vals_ave = coding_obj.encode(simulated_pulses_ave, trials)
        else:
            c_vals_pp = coding_obj.encode(simulated_pulses_pp)
            c_vals_ave = coding_obj.encode(simulated_pulses_ave)

        if rec_algo in ['linear', 'matchfilt']:
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
            axis[k].plot(c_vals_pp.transpose())

    if (shared_constants.debug):
        plt.show()
    return results



