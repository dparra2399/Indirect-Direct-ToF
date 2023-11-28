### Python imports
#### Library imports
import numpy as np
from scipy import stats
from IPython.core import debugger

from indirect_toflib import indirect_tof_utils
from research_utils import shared_constants
import matplotlib.pyplot as plt


def AddPoissonNoiseArr(Signal, trials=1000):
    new_size = (trials,) + Signal.shape
    rng = np.random.default_rng()
    return rng.poisson(lam=Signal, size=new_size).astype(Signal.dtype)


def IDTOF(Incident, DemodFs, depths, trials, dt=1, tbin_depth_res=None, src_incident=None):
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
                    src_depth = np.roll(src_incident[:, j], p)
                    photon_count = np.sum(src_depth)
                    if depths.shape[0] > 1:
                        axis[count].plot(np.transpose(cc_full), label='Sinosoid hist')
                        axis[count].plot(np.transpose(np.roll(Incident[:, j], p)), label='Ground truth')
                        axis[count].axvline(x=p, color='red', label='depth')
                        axis[count].set_title('Sinosoid Historgram with depth: ' + str(np.round(p*tbin_depth_res, decimals=2)) + ' / photon_count  : ' + str(np.round(photon_count, decimals=2)))
                        axis[count].legend()
                        count += 1
                    else:

                        axis.plot(np.transpose(cc_full), label='Sinosoid hist')
                        axis.plot(np.transpose(np.roll(Incident[:, j], p)), label='Ground truth')
                        axis.axvline(x=p, color='red', label='depth')
                        axis.set_title('Sinosoid Historgram with depth: ' + str(np.round(p*tbin_depth_res, decimals=2)) + ' / photon_count  : ' + str(np.round(photon_count, decimals=2)))
                        axis.legend()


        else:
            for l in range(0, depths.shape[0]):
                cc = np.roll(AddPoissonNoiseArr(Incident[:, j], trials), depths[l])

                measures[l, j, :] = np.inner(cc, demod)

    measures = measures * dt
    return measures

