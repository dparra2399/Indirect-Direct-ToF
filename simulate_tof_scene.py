# Python imports
import math
# Library imports
import numpy as np
from scipy import signal
from scipy import special
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace
mpl.use('qt5agg')
import CodingFunctions
import Utils
from Utils import plot
from toflib.coding import IdentityCoding
from toflib import tirf




def combined_and_indirect_mae(trials, depths, n_tbins, K, pAveAmbient, pAveSource, T, dMax, dt, freq, tau, meanBeta):


    gamma = 1./(pAveAmbient) #Camera gain

    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)

    kappas = np.sum(DemodFs, 0) * dt
    Ambient = pAveAmbient * kappas

    ModFs = Utils.ScaleIncident(ModFs, desiredArea=pAveSource)
    #ModFs = Utils.ScaleMod(ModFs, tau=tau, pAveSource=pAveSource)
    Incident = (gamma * meanBeta) * (T / tau) * (ModFs + Ambient)

    Measures = Utils.GetMeasurements(ModFs, DemodFs, dt=dt)

    NormMeasures = (Measures.transpose() - np.mean(Measures, axis=1)) / np.std(Measures, axis=1)
    NormMeasures = NormMeasures.transpose()

    mae_idtof = 0
    mae_itof = 0

    gt_depths = depths
    depths = np.round((depths / dMax) * n_tbins)
    for i in range(0, trials):
        ###DEPTH ESTIMATIONS
        measures_idtof = Utils.IDTOF(Incident, DemodFs, depths, dt=dt)
        measures_itof = Utils.ITOF(Incident, DemodFs, depths, dt=dt)

        norm_measurements_idtof = Utils.NormalizeMeasureVals(measures_idtof)
        norm_measurements_itof = Utils.NormalizeMeasureVals(measures_itof)

        decoded_depths_idtof = np.argmax(np.dot(NormMeasures, norm_measurements_idtof.transpose()), axis=0)
        decoded_depths_itof = np.argmax(np.dot(NormMeasures, norm_measurements_itof.transpose()), axis=0)

        decoded_depths_itof = decoded_depths_itof * dMax / n_tbins
        decoded_depths_idtof = decoded_depths_idtof * dMax / n_tbins
        (idtof, itof) = Utils.ComputeMetrics(gt_depths, decoded_depths_idtof, decoded_depths_itof)


        mae_idtof += idtof
        mae_itof += itof

    mae_itof = mae_itof / trials * 1000
    mae_idtof = mae_idtof / trials * 1000

    return (mae_idtof, mae_itof)


def direct_mae(trials, gt_depths, dMax, n_tbins, pAveAmbient, pAveSource,
               freq, tau, depth_padding, tbin_res, tbin_depth_res,
               t_domain, gt_tshifts, rec_algo, pw_factors):

    full_coding = IdentityCoding(n_maxres=n_tbins, account_irf=False, h_irf=None)

    # Create GT gaussian pulses for each coding. Different coding may use different pulse widths
    pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors * tbin_res, mu=gt_tshifts, t_domain=t_domain)
    pulses = pulses_list[0]

    curr_sbr = pAveSource / pAveAmbient
    pulses.set_sbr(curr_sbr)

    simulated_pulses = pulses.simulate_n_signal_photons(n_photons=pAveSource, n_mc_samples=trials)
    c_vals = full_coding.encode(simulated_pulses)
    # Estimate depths
    # with Timer("Decoding Time:"):
    decoded_depths = full_coding.maxgauss_peak_decoding(c_vals, gauss_sigma=pw_factors[0],
                                                       rec_algo_id=rec_algo) * tbin_depth_res

    # print("Decoded Depths: {}".format(decoded_depths))
    depth_errors = np.abs(decoded_depths - gt_depths[np.newaxis, :])
    depthwise_mae = depth_errors.mean(axis=0)
    mae = depthwise_mae.mean() * 1000
    return mae

def run_experiment(depths, n_tbins, K, T, dMax, fMax, tauMin, fSampling,
                   dt, freq, tau, meanBeta, pAveSourceList, pAveAmbientList, trials):

    (source_num,ambient_num) = pAveSourceList.shape

    SNR_IDTOF = np.zeros((source_num, ambient_num))
    SNR_ITOF = np.zeros((source_num, ambient_num))

    for x in range(0, source_num):
        for y in range(0, ambient_num):
            pAveSource = pAveSourceList[x, y]
            pAveAmbient = pAveAmbientList[x, y]

            (mae_idtof, mae_itof) = combined_and_indirect_mae(
                    trials=trials, depths=depths, n_tbins=n_tbins, K=K, pAveAmbient=pAveAmbient,
                    pAveSource=pAveSource, T=T,dMax=dMax, dt=dt, freq=freq, tau=tau, meanBeta=meanBeta)

            SNR_IDTOF[x, y] = mae_idtof
            SNR_ITOF[x, y] = mae_itof

    return (SNR_IDTOF, SNR_ITOF)




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
