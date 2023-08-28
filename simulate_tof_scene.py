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

import CodingFunctions
import Utils
from Utils import plot




def tof_scene(depths, n_tbins, K, pAveAmbient, pAveSource, T, dMax, fMax,
                   tauMin, fSampling, dt, freq, tau, meanBeta):


    gamma = 1. / (meanBeta * T * (pAveAmbient + pAveSource)) # Camera gain. Ensures all values are between 0-1

    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)

    kappas = np.sum(DemodFs, 0) * dt
    Ambient = pAveAmbient * kappas


    Incident = (gamma * meanBeta) * (T/tau) * (ModFs + Ambient)
    Incident = Utils.ScaleIncident(Incident, desiredArea=pAveSource)

    Measures = Utils.GetMeasurements(Incident, DemodFs)


    NormMeasures = (Measures.transpose() - np.mean(Measures, axis=1)) / np.std(Measures, axis=1)
    NormMeasures = NormMeasures.transpose()

    ###DEPTH ESTIMATIONS
    IDTOF = Utils.IDTOF(Incident, DemodFs)
    ITOF = Utils.ITOF(Incident, DemodFs)

    measures_idtof = IDTOF[depths.astype(int), :]
    measures_itof = ITOF[depths.astype(int), :]

    norm_measurements_idtof = Utils.NormalizeMeasureVals(measures_idtof)
    norm_measurements_itof = Utils.NormalizeMeasureVals(measures_itof)

    decoded_depths_idtof = np.argmax(np.dot(NormMeasures, norm_measurements_idtof.transpose()), axis=0)
    decoded_depths_itof = np.argmax(np.dot(NormMeasures, norm_measurements_itof.transpose()), axis=0)

    return (decoded_depths_idtof, decoded_depths_itof)


def run_experiment(depths, n_tbins, K, T, dMax, fMax, tauMin, fSampling,
                   dt, freq, tau, meanBeta, pAveSourceList, pAveAmbientList, trials):

    (source_num,ambient_num) = pAveSourceList.shape

    SNR_IDTOF = np.zeros((source_num, ambient_num))
    SNR_ITOF = np.zeros((source_num, ambient_num))

    for x in range(0, source_num):
        for y in range(0, ambient_num):
            pAveSourcePerPixel = pAveSourceList[x, y]
            pAveAmbientPerPixel = pAveAmbientList[x, y]

            mae_idtof = 0
            mae_itof = 0

            for i in range(0, trials):
                (decoded_depths_idtof, decoded_depths_itof) = tof_scene(
                    depths=depths, n_tbins=n_tbins, K=K, pAveAmbient=pAveAmbientPerPixel,
                    pAveSource=pAveSourcePerPixel, T=T,
                    dMax=dMax, fMax=fMax, tauMin=tauMin, fSampling=fSampling, dt=dt,
                    freq=freq, tau=tau,meanBeta=meanBeta)

                (idtof, itof) = Utils.ComputeMetrics(depths, decoded_depths_idtof, decoded_depths_itof)
                mae_idtof += idtof
                mae_itof += itof

            mae_itof = mae_itof / trials
            mae_idtof = mae_idtof / trials

            SNR_IDTOF[x, y] = mae_idtof
            SNR_ITOF[x, y] = mae_itof

    return (SNR_IDTOF, SNR_ITOF)



