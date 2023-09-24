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

breakpoint = debugger.set_trace

import CodingFunctions
import Utils
import FileUtils

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n_tbins = 1024
    K = 4
    #### Sensor parameters
    T = 0.1  # Integration time. Exposure time in seconds
    #### Coding function parameters
    speedOfLight = 299792458. * 1000.  # mm / sec
    fMax = 1e+7 # Maximum unambiguous repetition frequency (in Hz)
    tauMin = 1. / fMax
    dMax = 15
    fSampling = float(dMax) * fMax  # Sampling frequency of mod and demod functuion
    dt = tauMin / float(n_tbins)
    freq = fMax  # Fundamental frequency of modulation and demodulation functions
    tau = 1 / freq
    #### Scene parameters
    meanBeta = 1e-4  # Avg fraction of photons reflected from a scene points back to the detector
    #### Camera gain parameter


    #depth_img = np.load(
        #'./data/cvpr22_data/rendered_images/depth_images_240x320_nt-2000/bathroom-cycles-2_nr-240_nc-320_nt-2000_samples-2048_view-0.npy')
    #shape = depth_img.shape
    #depths = depth_img.ravel()
    #depths = depths[0:10]

    #depth_res = np.max(depths)
    #depths = (depths / depth_res) * (dMax - 1)

    # idtof_image = np.asarray((decoded_depths_idtof / (dMax - 1)) * depth_res).reshape(shape)
    # itof_image = np.asarray((decoded_depths_itof / (dMax - 1)) * depth_res).reshape(shape)

    depths = np.array([5.32, 6.78, 2.01, 7.68, 8.34])


    run_exp = 1
    exp_num = 21

    trials = 1000

    pAveSourcePerPixel = 1000
    pAveAmbientPerPixel = 1000

    grid = 25

    #pAveSourceList = np.linspace(10, 1000000, num=grid)
    #pAveAmbientList = np.linspace(100, 100000, num=grid)

    (min_signal_exp, max_signal_exp) = (1, 6)
    (min_amb_exp, max_amb_exp) = (2, 5)

    pAveSourceList = np.round(np.power(10, np.linspace(min_signal_exp, max_signal_exp, grid)))
    pAveAmbientList = np.power(10, np.linspace(min_amb_exp, max_amb_exp, grid))
    pAveSourceList, pAveAmbientList = np.meshgrid(pAveSourceList, pAveAmbientList)


    if run_exp:

        (SNR_IDTOF, SNR_ITOF) = simulate_tof_scene.run_experiment(
            depths=depths, n_tbins=n_tbins, K=K, pAveSourceList=pAveSourceList,
            pAveAmbientList=pAveAmbientList, T=T, dMax=dMax, fMax=fMax, tauMin=tauMin,
            fSampling=fSampling, dt=dt, freq=freq, tau=tau, meanBeta=meanBeta, trials=trials)

        FileUtils.WriteErrorsToFile(
            Coding='coscos', Experiment=exp_num, pAveSourceList=pAveSourceList, pAveAmbientList=pAveAmbientList,
            SNR_IDTOF=SNR_IDTOF, SNR_ITOF=SNR_ITOF, depths=depths, n_tbins=n_tbins, K=K,T=T,
            dMax=dMax, fMax=fMax,tauMin=tauMin, fSampling=fSampling, dt=dt, freq=freq, tau=tau,
            meanBeta=meanBeta, trials=trials)


    else:
        tic = time.perf_counter()
        (mae_idtof, mae_itof) = simulate_tof_scene.tof_scene_mae(
                trials=trials, depths=depths, n_tbins=n_tbins, K=K, pAveAmbient=pAveAmbientPerPixel,
                pAveSource=pAveSourcePerPixel, T=T,
                dMax=dMax, fMax=fMax, tauMin=tauMin, fSampling=fSampling, dt=dt,
                freq=freq, tau=tau, meanBeta=meanBeta)

        toc = time.perf_counter()

        print(f"MAE IDTOF: {mae_idtof: .3f},")
        print(f"MAE ITOF: {mae_itof: .3f},")

        print(f"Completed in {toc - tic:0.4f} seconds")

print('hello world')