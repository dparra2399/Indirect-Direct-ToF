# Python imports
import math
# Library imports
import numpy as np
from scipy import signal
from scipy import special
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
import time

import simulate_tof_scene
from toflib import tof_utils

breakpoint = debugger.set_trace

import CodingFunctions
import Utils
import FileUtils

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n_tbins = 1024
    K = 4
    T = 0.1  # Integration time. Exposure time in seconds
    speedOfLight = 299792458. * 1000.  # mm / sec
    rep_freq = 1e+7
    rep_tau = 1. / rep_freq
    dMax = tof_utils.time2depth(rep_tau)
    fSampling = float(dMax) * rep_freq  # Sampling frequency of mod and demod functuion
    dt = rep_tau / float(n_tbins)
    meanBeta = 1e-4  # Avg fraction of photons reflected from a scene points back to the detector

    ##DIRECT
    depth_padding = 0.02  # Skip the depths at the boundaries
    pw_factors = np.array([1, 1, 1, 8, 1])
    rec_algo = 'zncc'

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (tof_utils.calc_tof_domain_params(n_tbins, rep_tau=rep_tau))

    depths = np.array([5.32, 6.78, 2.01, 7.68, 8.34])
    gt_tshifts = tof_utils.depth2time(depths)

    run_exp = 0
    exp_num = 20

    trials = 1000

    pAveSourcePerPixel = 10
    pAveAmbientPerPixel = 1

    grid = 25

    (min_signal_exp, max_signal_exp) = (1, 6)
    (min_amb_exp, max_amb_exp) = (2, 5)

    pAveSourceList = np.round(np.power(10, np.linspace(min_signal_exp, max_signal_exp, grid)))
    pAveAmbientList = np.power(10, np.linspace(min_amb_exp, max_amb_exp, grid))
    pAveSourceList, pAveAmbientList = np.meshgrid(pAveSourceList, pAveAmbientList)



    if run_exp:

        (SNR_IDTOF, SNR_ITOF) = simulate_tof_scene.run_experiment(
            depths=depths, n_tbins=n_tbins, K=K, pAveSourceList=pAveSourceList,
            pAveAmbientList=pAveAmbientList, T=T, dMax=dMax, freq=rep_freq, tau=rep_tau,
            dt=dt, meanBeta=meanBeta, trials=trials)

        FileUtils.WriteErrorsToFile(
            Coding='coscos', Experiment=exp_num, pAveSourceList=pAveSourceList, pAveAmbientList=pAveAmbientList,
            SNR_IDTOF=SNR_IDTOF, SNR_ITOF=SNR_ITOF, depths=depths, n_tbins=n_tbins, K=K,T=T,
            dMax=dMax, dt=dt, freq=rep_freq, tau=rep_tau,
            meanBeta=meanBeta, trials=trials)


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