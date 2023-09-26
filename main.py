# Python imports
# Library imports
import numpy as np
from IPython.core import debugger
import time

import simulate_tof_scene
from direct_toflib import direct_tof_utils

breakpoint = debugger.set_trace

import FileUtils

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n_tbins = 1024
    K = 4
    T = 0.1  # Integration time. Exposure time in seconds
    speedOfLight = 299792458. * 1000.  # mm / sec
    rep_freq = 1e+7
    rep_tau = 1. / rep_freq
    dMax = direct_tof_utils.time2depth(rep_tau)
    fSampling = float(dMax) * rep_freq  # Sampling frequency of mod and demod functuion
    dt = rep_tau / float(n_tbins)
    meanBeta = 1 # Avg fraction of photons reflected from a scene points back to the detector

    ##DIRECT
    depth_padding = 0.02  # Skip the depths at the boundaries
    pw_factors = np.array([0.5])
    rec_algo = 'linear'

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (direct_tof_utils.calc_tof_domain_params(n_tbins, rep_tau=rep_tau))

    depths = np.array([5.32, 6.78, 2.01, 7.68, 8.34])
    gt_tshifts = direct_tof_utils.depth2time(depths)

    run_exp = 1
    exp_num = "low_snr"

    trials = 1000

    sourceExponent = 9
    ambientExponent = 6

    pAveSourcePerPixel = 10**sourceExponent
    pAveAmbientPerPixel = 10*ambientExponent

    grid = 20

    (min_signal_exp, max_signal_exp) = (8, 12)
    (min_amb_exp, max_amb_exp) = (7, 9)

    pAveSourceList = np.round(np.power(10, np.linspace(min_signal_exp, max_signal_exp, grid)))
    pAveAmbientList = np.power(10, np.linspace(min_amb_exp, max_amb_exp, grid))
    pAveSourceList, pAveAmbientList = np.meshgrid(pAveSourceList, pAveAmbientList)



    if run_exp:

        tic = time.perf_counter()
        (depths_errors_idtof, depth_errors_itof, depths_errors_dtof) = simulate_tof_scene.run_experiment(
            trials=trials, depths=depths, dMax=dMax, n_tbins=n_tbins, pAveAmbientList=pAveAmbientList,
            pAveSourceList=pAveSourceList, freq=rep_freq, tau=rep_tau,
            depth_padding=depth_padding, tbin_res=tbin_res,
            tbin_depth_res=tbin_depth_res, t_domain=t_domain,
            gt_tshifts=gt_tshifts, rec_algo=rec_algo, pw_factors=pw_factors,
            K=K, T=T, dt=dt, meanBeta=meanBeta)

        toc = time.perf_counter()
        print(f"Completed in {toc - tic:0.4f} seconds")

        FileUtils.WriteErrorsToFile(
            Experiment=exp_num,
            trials=trials, depths=depths, dMax=dMax, n_tbins=n_tbins,
            pAveAmbientList=pAveAmbientList,
            pAveSourceList=pAveSourceList, freq=rep_freq, tau=rep_tau,
            depth_padding=depth_padding, tbin_res=tbin_res,
            tbin_depth_res=tbin_depth_res, t_domain=t_domain,
            gt_tshifts=gt_tshifts, rec_algo=rec_algo, pw_factors=pw_factors,
            K=K, T=T, dt=dt, meanBeta=meanBeta, mae_idtof=depths_errors_idtof,
             mae_itof=depth_errors_itof, mae_dtof=depths_errors_dtof)


    else:
        tic = time.perf_counter()
        (mae_idtof, mae_itof, mae_dtof) = simulate_tof_scene.all_tof_mae(
                trials=trials, depths=depths, dMax=dMax, n_tbins=n_tbins, pAveAmbient=pAveAmbientPerPixel,
                pAveSource=pAveSourcePerPixel, freq=rep_freq, tau=rep_tau,
                depth_padding=depth_padding, tbin_res=tbin_res,
                tbin_depth_res=tbin_depth_res, t_domain=t_domain,
                gt_tshifts=gt_tshifts, rec_algo=rec_algo, pw_factors=pw_factors,
                K=K, T=T, dt=dt, meanBeta=meanBeta)


        toc = time.perf_counter()

        print(f"MAE IDTOF: {mae_idtof: .3f},")
        print(f"MAE ITOF: {mae_itof: .3f},")
        print(f"MAE DTOF: {mae_dtof: .3f}")

        print(f"Completed in {toc - tic:0.4f} seconds")

print('hello world')