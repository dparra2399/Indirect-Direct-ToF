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




def tof_scene_mae(trials, depths, n_tbins, K, pAveAmbient, pAveSource, T, dMax, fMax,
                   tauMin, fSampling, dt, freq, tau, meanBeta):


    gamma = 1./(pAveAmbient) #Camera gain


    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)

    kappas = np.sum(DemodFs, 0) * dt
    Ambient = pAveAmbient * kappas

    #ModFs = Utils.ScaleIncident(ModFs, desiredArea=pAveSource)
    ModFs = Utils.ScaleMod(ModFs, tau=tauMin, pAveSource=pAveSource)
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

        #decoded_depths_itof = decoded_depths_itof * dMax / n_tbins
        #decoded_depths_idtof = decoded_depths_idtof * dMax / n_tbins
        (idtof, itof) = Utils.ComputeMetrics(depths, decoded_depths_idtof, decoded_depths_itof)


        mae_idtof += idtof
        mae_itof += itof

    mae_itof = mae_itof / trials
    mae_idtof = mae_idtof / trials

    return (mae_idtof, mae_itof)


def run_experiment(depths, n_tbins, K, T, dMax, fMax, tauMin, fSampling,
                   dt, freq, tau, meanBeta, pAveSourceList, pAveAmbientList, trials):

    (source_num,ambient_num) = pAveSourceList.shape

    SNR_IDTOF = np.zeros((source_num, ambient_num))
    SNR_ITOF = np.zeros((source_num, ambient_num))

    for x in range(0, source_num):
        for y in range(0, ambient_num):
            pAveSourcePerPixel = pAveSourceList[x, y]
            pAveAmbientPerPixel = pAveAmbientList[x, y]

            (mae_idtof, mae_itof) = tof_scene_mae(
                trials=trials, depths=depths, n_tbins=n_tbins, K=K, pAveAmbient=pAveAmbientPerPixel,
                pAveSource=pAveSourcePerPixel, T=T,
                dMax=dMax, fMax=fMax, tauMin=tauMin, fSampling=fSampling, dt=dt,
                freq=freq, tau=tau, meanBeta=meanBeta)

            SNR_IDTOF[x, y] = mae_idtof
            SNR_ITOF[x, y] = mae_itof

    return (SNR_IDTOF, SNR_ITOF)



