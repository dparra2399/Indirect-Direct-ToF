### Python imports
#### Library imports
import numpy as np
from scipy import stats
from IPython.core import debugger
from research_utils import shared_constants
import matplotlib.pyplot as plt


def AddPoissonNoiseArr(Signal, trials=1000):
    new_size = (trials,) + Signal.shape
    rng = np.random.default_rng()
    return rng.poisson(lam=Signal, size=new_size).astype(Signal.dtype)


def IDTOF(Incident, DemodFs, depths, trials, dt=1):
    (n_tbins, K) = Incident.shape

    measures = np.zeros((depths.shape[0], K, trials))

    if (shared_constants.debug):
        assert trials == 1, 'one sample needed fro debugging'
        measures = np.zeros((n_tbins, K))
        figure, axis = plt.subplots(depths.shape[0])
        count = 0

    depths = depths.astype(int)
    for j in range(0, K):
        demod = DemodFs[:, j]

        if (shared_constants.debug):
            count = 0
            for p in range(0, n_tbins):
                cc_full = np.roll(AddPoissonNoiseArr(Incident[:, j], 1), p)
                measures[p, j] = np.inner(cc_full, demod)

                if p in depths and j == 0:
                    if depths.shape[0] > 1:
                        axis[count].plot(np.transpose(cc_full))
                        axis[count].plot(np.transpose(np.roll(Incident[:, j], p)))
                        count += 1
                    else:
                        axis.plot(np.transpose(cc_full))
                        axis.plot(np.transpose(np.roll(Incident[:, j], p)))


        else:
            for l in range(0, depths.shape[0]):
                cc = np.roll(AddPoissonNoiseArr(Incident[:, j], trials), depths[l])

                measures[l, j, :] = np.inner(cc, demod)

    measures = measures * dt
    return measures


def pulses_idtof(pulses, DemodFs, depths, trials, dt=1):
    (n_tbins, K) = DemodFs.shape
    measures = np.zeros((depths.shape[0], K, trials))

    depths = depths.astype(int)
    for j in range(0, K):
        demod = DemodFs[:, j]
        for l in range(0, depths.shape[0]):
            pulse = AddPoissonNoiseArr(pulses[:, depths[l]], trials)
            measures[l, j, :] = np.inner(pulse, demod)

    measures = measures * dt
    return measures

def GetPulseMeasurements(Clean_pulse, DemodFs, dt=1):
    (n_tbins, K) = DemodFs.shape
    measures = np.zeros((n_tbins, K))

    for j in range(0, K):
        demod = DemodFs[:, j]
        for l in range(0, n_tbins):
            pulse = Clean_pulse[:, l]
            measures[l, j] = np.inner(pulse, demod)

    measures = measures * dt
    return measures
