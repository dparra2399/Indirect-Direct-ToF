# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace
mpl.use('qt5agg')
from combined_indirect_tof import combined_indirect_utils, CodingFunctions
from direct_toflib import tirf
from direct_toflib.direct_tof_utils import time2depth
import math


def combined_and_indirect_mae(trials, depths, n_tbins, K, pAveAmbient, pAveSource, T, dMax, dt, freq, tau, meanBeta):

    #gamma = 1. / (meanBeta * T * (pAveAmbient + pAveSource))  # Camera gain. Ensures all values are between 0-1.

    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)

    kappas = np.sum(DemodFs, 0) * dt
    Ambient = tau * pAveAmbient * kappas

    ModFs = combined_indirect_utils.ScaleMod(ModFs, tau=tau, pAveSource=pAveSource)
    Incident = (ModFs + Ambient)

    Measures = combined_indirect_utils.GetMeasurements(ModFs, DemodFs, dt=dt)

    NormMeasures = (Measures.transpose() - np.mean(Measures, axis=1)) / np.std(Measures, axis=1)
    NormMeasures = NormMeasures.transpose()

    mae_idtof = 0
    mae_itof = 0

    gt_depths = depths
    depths = np.round((depths / dMax) * n_tbins)
    for i in range(0, trials):
        ###DEPTH ESTIMATIONS
        measures_idtof = combined_indirect_utils.IDTOF(Incident, DemodFs, depths, dt=dt)
        measures_itof = combined_indirect_utils.ITOF(Incident, DemodFs, depths, dt=dt)

        norm_measurements_idtof = combined_indirect_utils.NormalizeMeasureVals(measures_idtof)
        norm_measurements_itof = combined_indirect_utils.NormalizeMeasureVals(measures_itof)

        decoded_depths_idtof = np.argmax(np.dot(NormMeasures, norm_measurements_idtof.transpose()), axis=0)
        decoded_depths_itof = np.argmax(np.dot(NormMeasures, norm_measurements_itof.transpose()), axis=0)

        decoded_depths_itof = decoded_depths_itof * dMax / n_tbins
        decoded_depths_idtof = decoded_depths_idtof * dMax / n_tbins
        (idtof, itof) = combined_indirect_utils.ComputeMetrics(gt_depths, decoded_depths_idtof, decoded_depths_itof)


        mae_idtof += idtof
        mae_itof += itof

    mae_itof = mae_itof / trials * 1000
    mae_idtof = mae_idtof / trials * 1000

    return (mae_idtof, mae_itof)


def direct_mae(trials, gt_depths, dMax, n_tbins, pAveAmbient, pAveSource,
               freq, tau, depth_padding, tbin_res, tbin_depth_res,
               t_domain, gt_tshifts, rec_algo, pw_factors, peak_power=None):

    # Create GT gaussian pulses for each coding. Different coding may use different pulse widths
    pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors * tbin_res, mu=gt_tshifts, t_domain=t_domain)
    pulses = pulses_list[0]

    curr_sbr = pAveSource / pAveAmbient
    n_photons = tau * pAveSource #total photons
    pulses.set_sbr(curr_sbr)

    simulated_pulses = pulses.simulate_n_signal_photons(n_photons=n_photons, n_mc_samples=trials)
    # Estimate depths
    # with Timer("Decoding Time:"):

    decoded_depths = np.argmax(simulated_pulses, axis=-1) * tbin_depth_res
    depth_errors = np.abs(decoded_depths - gt_depths[np.newaxis, :])
    depthwise_mae = depth_errors.mean(axis=0)
    mae = depthwise_mae.mean() * 1000
    return mae

def run_experiment(trials, depths, dMax, n_tbins, pAveAmbientList,
                pAveSourceList, freq, tau,
                depth_padding, tbin_res, tbin_depth_res,
                t_domain, gt_tshifts, rec_algo, pw_factors,
                K, T, dt, meanBeta):

    (source_num,ambient_num) = pAveSourceList.shape

    depths_errors_idtof = np.zeros((source_num, ambient_num))
    depth_errors_itof = np.zeros((source_num, ambient_num))
    depths_errors_dtof = np.zeros((source_num, ambient_num))

    for x in range(0, source_num):
        for y in range(0, ambient_num):
            pAveSource = pAveSourceList[x, y]
            pAveAmbient = pAveAmbientList[x, y]

            (mae_idtof, mae_itof, mae_dtof) = all_tof_mae(
                trials=trials, depths=depths, dMax=dMax, n_tbins=n_tbins, pAveAmbient=pAveAmbient,
                pAveSource=pAveSource, freq=freq, tau=tau,
                depth_padding=depth_padding, tbin_res=tbin_res,
                tbin_depth_res=tbin_depth_res, t_domain=t_domain,
                gt_tshifts=gt_tshifts, rec_algo=rec_algo, pw_factors=pw_factors,
                K=K, T=T, dt=dt, meanBeta=meanBeta)


            depths_errors_idtof[x, y] = mae_idtof
            depth_errors_itof[x, y] = mae_itof
            depths_errors_dtof[x, y] = mae_dtof

    return (depths_errors_idtof, depth_errors_itof, depths_errors_dtof)




def all_tof_mae(trials, depths, dMax, n_tbins, pAveAmbient,
                pAveSource, freq, tau,
                depth_padding, tbin_res, tbin_depth_res,
                t_domain, gt_tshifts, rec_algo, pw_factors,
                K, T, dt, meanBeta):

    (mae_idtof, mae_itof) = combined_and_indirect_mae(
        trials=trials, depths=depths, n_tbins=n_tbins, K=K, pAveAmbient=pAveAmbient,
        pAveSource=pAveSource, T=T,dMax=dMax, dt=dt, freq=freq, tau=tau, meanBeta=meanBeta)

    mae_dtof = direct_mae(trials=trials,gt_depths=depths, dMax=dMax, n_tbins=n_tbins, pAveAmbient=pAveAmbient,
                pAveSource=pAveSource, freq=freq, tau=tau,
                depth_padding=depth_padding, tbin_res=tbin_res,
                tbin_depth_res=tbin_depth_res, t_domain=t_domain,
                gt_tshifts=gt_tshifts, rec_algo=rec_algo, pw_factors=pw_factors)



    return(mae_idtof, mae_itof, mae_dtof)
