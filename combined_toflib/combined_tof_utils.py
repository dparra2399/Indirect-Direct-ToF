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


def IDTOF(Incident, DemodFs, depths, trials, gated=False, dt=1, tbin_depth_res=None, src_incident=None):
    (n_tbins, K) = Incident.shape

    measures = np.zeros((depths.shape[0], K, trials))

    depths = depths.astype(int)
    for j in range(0, K):
        demod = DemodFs[:, j]
        for l in range(0, depths.shape[0]):
            #for l in range(0, n_tbins):
            convolve = 0
            if gated == True:
                splitted_gates = gatedHam(demod)
                for q in range(0, splitted_gates.shape[-1]):
                    gate = splitted_gates[:, q]
                    cc = np.roll(AddPoissonNoiseArr(Incident[:, j], trials), depths[l])
                    convolve += np.inner(cc, gate)
            else:
                cc = np.roll(AddPoissonNoiseArr(Incident[:, j], trials), depths[l])
                convolve = np.inner(cc, demod)

            measures[l, j, :] = convolve

    measures = measures * dt
    return measures

# tmp = np.zeros((n_tbins, 1))
# for i in range(0, K):
#     demod = DemodFs[:, i]
#     splitted_gates = gatedHam(demod)
#     tmp = np.concatenate((tmp, splitted_gates), axis=-1)

def gatedHam(demod):
    n_tbins = demod.shape[0]
    gates = np.zeros((n_tbins, 1))
    squares = split_into_indices(demod)
    for j in range(len(squares)):
        pair = squares[j]
        single_square = np.zeros((n_tbins, 1))
        single_square[pair[0]:pair[1]+1, :] = 1
        gates = np.concatenate((gates, single_square), axis=-1)

    gates = gates[:, 1:]
    return gates





def split_into_indices(square_array):
    indices = []
    start_index = None
    for i, num in enumerate(square_array):
        if num == 1:
            if start_index is None:
                start_index = i
        elif num == 0 and start_index is not None:
            indices.append((start_index, i - 1))
            start_index = None
    if start_index is not None:
        indices.append((start_index, len(square_array) - 1))
    return indices

    #
    # if (shared_constants.debug):
    #     assert trials == 1, 'one sample needed fro debugging'
    #     measures = np.zeros((n_tbins, K))
    #     figure, axis = plt.subplots(depths.shape[0])
    #     count = 0
    #
# if (shared_constants.debug):
#     count = 0
#     for p in range(0, n_tbins):
#         cc_full = np.roll(AddPoissonNoiseArr(Incident[:, j], 1), p)
#         measures[p, j] = np.inner(cc_full, demod)
#
#         if p in depths and j == 0:
#             src_depth = np.roll(src_incident[:, j], p)
#             photon_count = np.sum(src_depth)
#             if depths.shape[0] > 1:
#                 axis[count].plot(np.transpose(cc_full), label='Sinosoid hist')
#                 axis[count].plot(np.transpose(np.roll(Incident[:, j], p)), label='Ground truth')
#                 axis[count].axvline(x=p, color='red', label='depth')
#                 axis[count].set_title('Sinosoid Historgram with depth: ' + str(
#                     np.round(p * tbin_depth_res, decimals=2)) + ' / photon_count  : ' + str(
#                     np.round(photon_count, decimals=2)))
#                 axis[count].legend()
#                 count += 1
#             else:
#
#                 axis.plot(np.transpose(cc_full), label='Sinosoid hist')
#                 axis.plot(np.transpose(np.roll(Incident[:, j], p)), label='Ground truth')
#                 axis.axvline(x=p, color='red', label='depth')
#                 axis.set_title('Sinosoid Historgram with depth: ' + str(
#                     np.round(p * tbin_depth_res, decimals=2)) + ' / photon_count  : ' + str(
#                     np.round(photon_count, decimals=2)))
#                 axis.legend()
