### Python imports
#### Library imports
import numpy as np
from scipy import stats
from IPython.core import debugger
import matplotlib.pyplot as plt


def AddPoissonNoiseArr(Signal, trials=1000):
    new_size = (trials,) + Signal.shape
    rng = np.random.default_rng()
    return rng.poisson(lam=Signal, size=new_size).astype(Signal.dtype)


def IDTOF(Incident, DemodFs, depths, trials, dt=1):
    (n_tbins, K) = Incident.shape

    measures = np.zeros((depths.shape[0], K, trials))
    depths = depths.astype(int)
    for j in range(0, K):
        demod = DemodFs[:, j]

        for l in range(0, depths.shape[0]):
            cc = np.roll(AddPoissonNoiseArr(Incident[:, j], trials), depths[l])
            measures[l, j, :] = np.inner(cc, demod)

    measures = measures * dt
    return measures


def pulses_idtof(pulses, DemodFs, trials, dt=1):
    (n_tbins, K) = DemodFs.shape
    measures = np.zeros((pulses.shape[1], K, trials))

    for j in range(0, K):
        demod = DemodFs[:, j]
        for l in range(0, pulses.shape[1]):
            pulse = pulses[:, l, :]
            measures[l, j, :] = np.inner(pulse, demod)

    measures = measures * dt
    return measures

def GetPulseMeasurements(Clean_pulse, DemodFs, dt=1):
    (n_tbins, K) = DemodFs.shape
    measures = np.zeros((n_tbins, K))

    for j in range(0, K):
        demod = DemodFs[:, j]
        for l in range(0, n_tbins):
            pulse = np.roll(Clean_pulse, l)
            measures[l, j] = np.inner(pulse, demod)

    measures = measures * dt
    return measures
