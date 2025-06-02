### Python imports
#### Library imports
import numpy as np
from felipe_utils.tof_utils_felipe import *
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


def simulate_average_photons_n_cycles(light_source, total_photons, sbr):
    incident = np.zeros(light_source.shape)
    n_tbins = light_source.shape[0]

    if sbr == 0:
        total_amb_photons = 0
    else:
        total_amb_photons = total_photons / sbr
    scaled_modfs = np.copy(light_source)
    for i in range(0, light_source.shape[-1]):
        scaled_modfs[:, i] *= (total_photons / np.sum(light_source[:, i]))
        incident[:, i] = (scaled_modfs[:, i] + (total_amb_photons / n_tbins))
    return incident


def phase_shifted(light_source, depths, tbin_depth_res):
    shifted_modfs = np.zeros((depths.shape[0], light_source.shape[1], light_source.shape[0]))
    for d in range(0, depths.shape[0]):
        for i in range(0, light_source.shape[-1]):
            shifted_modfs[d, i, :] = np.roll(light_source[:, i], int(depths[d] / tbin_depth_res))
    return shifted_modfs

def zncc_reconstruction(intensities, corrs):
    norm_int = zero_norm_t(intensities, axis=-1)
    zero_norm_corrs = zero_norm_t(corrs, axis=-1)
    return np.matmul(zero_norm_corrs, norm_int[..., np.newaxis]).squeeze(-1)

def ncc_reconstruction(intensities, corrs):
    norm_int = norm_t(intensities, axis=-1)
    norm_corrs = norm_t(corrs, axis=-1)
    return np.matmul(norm_corrs, norm_int[..., np.newaxis]).squeeze(-1)

def compute_scale(beta, alpha, beta_q, alpha_q):
	return (float(beta) - float(alpha)) / (float(beta_q) - float(alpha_q))
	# return ((beta) - (alpha)) / ((beta_q) - (alpha_q))

def compute_zero_point(scale, alpha, alpha_q):
	## Cast everything to float first to make sure that we dont input torch.tensors to zero point
	return round(-1*((float(alpha)/float(scale)) - float(alpha_q)))


def quantize_qint8_numpy(X, X_range=None):
	if(X_range is None):
		(X_min, X_max) = (X.min(), X.max())
	else:
		X_min = X_range[0]
		X_max = X_range[1]
		assert(X_min <= X.min()), "minimum should be contained in range"
		assert(X_max >= X.max()), "maximum should be contained in range"
	print("manual min: {}".format(X_min))
	print("manual max: {}".format(X_max))
	(min_q, max_q) = (-128, 127)
	scale = compute_scale(X_max, X_min, max_q, min_q)
	zero_point = compute_zero_point(scale, X_min, min_q)
	qX = (X / scale) + zero_point
	qX_int8  = np.round(qX).astype(np.int8)
	return  (qX_int8, scale, zero_point)