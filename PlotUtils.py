### Python imports
#### Library imports
import numpy as np
from scipy import stats
from IPython.core import debugger
import matplotlib.pyplot as plt

from indirect_toflib.indirect_tof_utils import AddPoissonNoiseLam
from combined_toflib.combined_tof_utils import AddPoissonNoiseArr
breakpoint = debugger.set_trace

def FULL_IDTOF(Incident, DemodFs, trials=1, dt=1):
    (n_tbins, K) = Incident.shape

    measures = np.zeros((n_tbins, K, trials))
    for j in range(0, K):
        demod = DemodFs[:, j]

        for l in range(0, n_tbins):
            cc = np.roll(AddPoissonNoiseArr(Incident[:, j], trials), l)
            measures[l, j, :] = np.inner(cc, demod)

    measures = measures * dt
    return measures


def FULL_pulses_idtof(pulses, DemodFs, depths, trials=1, dt=1):
    (n_tbins, K) = DemodFs.shape
    measures = np.zeros((n_tbins, depths.shape[0], trials))

    depths = depths.astype(int)
    demod = DemodFs[:, 3]
    for j in range(0, depths.shape[0]):
        for l in range(0, n_tbins):
            pulse = np.roll(pulses[j, :], l)
            measures[l, j, :] = np.inner(pulse, demod)

    measures = np.squeeze(measures * dt)
    return measures



def FULL_ITOF(Incident, DemodFs, depths, trials=1, dt=1):
    (n_tbins, K) = Incident.shape
    measures = np.zeros((n_tbins, K, trials))
    depths = depths.astype(int)

    for j in range(0, K):
        demod = DemodFs[:, j]

        for l in range(0, n_tbins):
            cc = np.roll(Incident[:, j], l)
            measures[l, j, :] = AddPoissonNoiseLam(np.inner(cc, demod), trials)

    measures = measures * dt
    return measures


def plot_signals(pulses, sin):
    plot_pulses = np.transpose(np.squeeze(pulses[0,...]))
    plt.plot(plot_pulses)
    plt.plot(np.squeeze(AddPoissonNoiseArr(sin, trials=1)))
    plt.show()