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



def quantize_any_bits_numpy(X, bits, X_range=None):
    """
    Quantize `X` into signed integers of `bits` bits.
    Returns (qX_int, scale, zero_point).
    """
    if X_range is None:
        x_min, x_max = X.min(), X.max()
    else:
        x_min, x_max = X_range[0], X_range[1]
        assert x_min <= X.min(), "X_range[0] must be ≤ X.min()"
        assert x_max >= X.max(), "X_range[1] must be ≥ X.max()"
    min_q = - (2 ** (bits - 1))
    max_q =   2 ** (bits - 1) - 1
    scale = compute_scale(x_max, x_min, max_q, min_q)
    zero_point = compute_zero_point(scale, x_min, min_q)
    qX = (X / scale) + zero_point
    qX_clamped = np.clip(np.round(qX), min_q, max_q)
    if bits <= 8:
        out_dtype = np.int8
    elif bits <= 16:
        out_dtype = np.int16
    elif bits <= 32:
        out_dtype = np.int32
    else:
        out_dtype = np.int64
    qX_int = qX_clamped.astype(out_dtype)
    return qX_int, scale, zero_point

"""
Not my code from CHATGPT 
"""
def reconstruct_and_get_code_global_top_n(coding_matrix, num_coefficients_to_keep):
    """
    Performs Fourier transform on the entire coding matrix and keeps only the
    largest N magnitude coefficients across all columns for reconstruction.

    Args:
        coding_matrix (numpy.ndarray): A 2D numpy array representing the coding matrix (rows, cols).
        num_coefficients_to_keep (int): The total number of largest magnitude coefficients to keep across the entire matrix.

    Returns:
        numpy.ndarray: The reconstructed coding matrix.
    """
    fourier_coeffs = np.fft.fft(coding_matrix, axis=0)
    magnitudes = np.abs(fourier_coeffs)

    # Flatten the magnitude array to find the top N indices globally
    magnitudes_flat = magnitudes.flatten()
    ind_sorted_flat = np.argsort(magnitudes_flat)[::-1]
    top_n_indices_flat = ind_sorted_flat[:num_coefficients_to_keep]

    # Create a mask for the Fourier coefficients
    mask = np.zeros_like(fourier_coeffs, dtype=bool)
    rows, cols = coding_matrix.shape
    row_indices, col_indices = np.unravel_index(top_n_indices_flat, (rows, cols))

    # Set the mask to True for the top N coefficients
    mask[row_indices, col_indices] = True

    # Apply the mask to zero out the smaller coefficients
    modified_fourier_coeffs = np.where(mask, fourier_coeffs, 0)

    # Perform inverse Fourier transform
    reconstructed_matrix = np.fft.ifft(modified_fourier_coeffs, axis=0).real
    return reconstructed_matrix

