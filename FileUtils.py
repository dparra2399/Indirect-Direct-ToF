# Python imports
import math
# Library imports
import numpy as np
from scipy import signal
from scipy import special
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger

import simulate_tof_scene

breakpoint = debugger.set_trace

import CodingFunctions
import Utils
import os.path


def WriteErrorsToFile(Experiment, Coding, pAveSourceList, pAveAmbientList, SNR_IDTOF, SNR_ITOF, depths, n_tbins,
                      K, T, dMax, dt, freq, tau, meanBeta, trials):

    filename = 'ntbins_{}_k_{}_coding_{}_monte_{}_exp_{}.npz'.format(n_tbins, K, Coding, trials,  Experiment)
    outfile = './data/' + filename

    if os.path.isfile(outfile):
        print("Filename {} exits".format(filename))
        #exit(0)

    np.savez(outfile, Coding='coscos', pAveSourceList=pAveSourceList, pAveAmbientList=pAveAmbientList,
            SNR_IDTOF=SNR_IDTOF, SNR_ITOF=SNR_ITOF, depths=depths, n_tbins=n_tbins, K=K,T=T,
            dMax=dMax, dt=dt, freq=freq, tau=tau,
            meanBeta=meanBeta, trials=trials)
