### Python imports
#### Library imports
import numpy as np
from IPython.core import debugger

breakpoint = debugger.set_trace

def gaussian_irf(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def calculate_ambient(n_tbins, ave_ambient, tau, dt=None):
    eTotal = tau * ave_ambient
    base = np.ones(n_tbins)
    if dt is None: dt = tau/n_tbins
    oldArea = np.sum(base) * dt
    amb = base * eTotal / oldArea
    return amb


def normalize_measure_vals(b_vals, axis=-1):
    norm_bvals = (b_vals - np.mean(b_vals, axis=axis, keepdims=True)) / np.std(b_vals, axis=axis, keepdims=True)
    return norm_bvals


def poisson_noise_array(Signal, trials=1000):
    new_size = (trials,) + Signal.shape
    rng = np.random.default_rng()
    return rng.poisson(lam=Signal, size=new_size).astype(Signal.dtype)


def compute_metrics(depths, decoded_depths):
    errors = np.abs(decoded_depths - depths[np.newaxis, :])
    mae = np.mean(np.mean(errors, axis=0))
    return mae


def gated_ham(demod):
    n_tbins = demod.shape[0]
    gates = np.zeros((n_tbins, 1))
    squares = split_into_indices(demod)
    for j in range(len(squares)):
        pair = squares[j]
        single_square = np.zeros((n_tbins, 1))
        single_square[pair[0]:pair[1] + 1, :] = 1
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
