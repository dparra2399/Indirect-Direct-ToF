# Python imports
# Library imports
import time

import numpy as np
from IPython.core import debugger

from felipe_utils import tof_utils_felipe
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse
from plot_figures.plot_utils import *
from felipe_utils.CodingFunctionsFelipe import *
from scipy.interpolate import interp1d

from spad_toflib.spad_tof_utils import decompose_ham_codes

#matplotlib.use('QTkAgg')
breakpoint = debugger.set_trace


def get_voltage_function(mhz, voltage, illum_type, n_tbins=None):
    function = np.genfromtxt(f'/Users/davidparra/PycharmProjects/py-gated-camera/voltage_functions/{illum_type}_{mhz}mhz_{voltage}v.csv',delimiter=',')[:, 1]

    modfs = function[2:]
    if illum_type == 'pulse':
        modfs[150:600] = 0
        modfs = np.roll(modfs, -np.argmax(modfs), axis=0)
    elif illum_type == 'square':
        modfs[180:600] = 0

    if n_tbins is not None:
        f = interp1d(np.linspace(0, 1, len(modfs)), modfs, kind='cubic')
        modfs = f(np.linspace(0, 1, n_tbins))

    modfs /= np.sum(modfs, keepdims=True)
    return modfs

n_tbins = 1024
n_gates = 8
photon_count = 10
sbr = 5.0
trials = 1_000


#depth = np.random.randint(200, 701)
depths = np.arange(250, 750, 10)
# upper bound is exclusive
rng = np.random.default_rng()  # optional seed for reproducibility

photon_count_gated = photon_count // n_gates
illums = np.zeros((depths.shape[0], n_tbins))
for i, depth in enumerate(depths):
    illum = np.roll(gaussian_pulse(np.arange(n_tbins), 0, 50, circ_shifted=True), depth)
    #illum = np.roll(get_voltage_function(10, 10, 'pulse', n_tbins=n_tbins), depth)
    illums[i, :] = (illum * (photon_count_gated / np.sum(illum))) + ((photon_count_gated / sbr) / n_tbins)


coding_matrix = np.kron(np.eye(n_gates), np.ones((1, n_tbins // n_gates)))
irf = gaussian_pulse(np.arange(n_tbins), 0, 50, circ_shifted=True)
#irf = get_voltage_function(10, 10, 'pulse', n_tbins=n_tbins)
irf = irf.squeeze()
irf = irf[..., np.newaxis]
coding_matrix = np.fft.ifft(np.fft.fft(irf, axis=0).conj() * np.fft.fft(np.transpose(coding_matrix), axis=0), axis=0).real

#illum = rng.poisson(illums, size=(trials, illum.shape[0]))
clean_coded_vals = np.matmul(np.transpose(coding_matrix), illums[..., np.newaxis]).squeeze(-1)

coded_vals = rng.poisson(clean_coded_vals, size=(trials, clean_coded_vals.shape[0], clean_coded_vals.shape[1]))


norm_coding_matrix = tof_utils_felipe.zero_norm_t(coding_matrix)

norm_coded_vals = tof_utils_felipe.zero_norm_t(coded_vals)

zncc = np.matmul(norm_coding_matrix, norm_coded_vals[..., np.newaxis]).squeeze(-1)

decoded_depth = np.argmax(zncc, axis=-1)


(modfs2, demodfs) = GetHamK3(n_tbins)
#modfs2 = np.tile(get_voltage_function(10, 10, 'square', n_tbins=n_tbins)[..., np.newaxis], 3)
demodfs_np, demodfs_arr = decompose_ham_codes(demodfs)
#modfs2 = np.repeat(modfs2, 4, axis=1)[:, :demodfs.shape[-1]]
modfs_tmp = np.copy(np.repeat(modfs2, 4, axis=1)[:, :demodfs_np.shape[-1]])

photon_count_ham = photon_count // demodfs_np.shape[-1]

modfs = np.zeros((depths.shape[0], modfs_tmp.shape[0], modfs_tmp.shape[1]))
for j, depth in enumerate(depths):
    for i in range(modfs.shape[-1]):
        modf = (modfs_tmp[:, i]  * (photon_count_ham / np.sum(modfs_tmp[:, i] ))) + ((photon_count_ham / sbr) / n_tbins)
        modfs[j, :, i]  = np.roll(modf , depth)

#modfs = rng.poisson(modfs, size=(trials, modfs.shape[0], modfs.shape[1]))
clean_brightness_vals = np.zeros((depths.shape[0], modfs.shape[-1]))
for j in range(modfs.shape[-1]):
    clean_brightness_vals[:, j] = np.inner(demodfs_np[:, j], modfs[:, :, j])


correlations = np.zeros_like(modfs2)
for d in range(demodfs.shape[-1]):
    for n in range(n_tbins):
        value = np.inner(demodfs[:, d], np.roll(modfs2[:, d], n))
        correlations[n, d] = value

original_K = correlations.shape[-1]
temp = np.zeros((clean_brightness_vals.shape[0], original_K))
counter = 0
for j, item in enumerate(demodfs_arr):
    for i in range(item.shape[-1]):
        temp[:, j] += clean_brightness_vals[:, counter]
        counter += 1

clean_brightness_vals = temp

brightness_vals = rng.poisson(clean_brightness_vals, size=(trials, clean_brightness_vals.shape[0], clean_brightness_vals.shape[1]))

norm_correlations = tof_utils_felipe.zero_norm_t(correlations)

norm_brightness_vals = tof_utils_felipe.zero_norm_t(brightness_vals)

zncc_ham = np.matmul(norm_correlations, norm_brightness_vals[..., np.newaxis]).squeeze(-1)

decoded_depth_ham = np.argmax(zncc_ham, axis=-1)

print(f'Coarse: \n\t MAE: {np.mean(np.abs(decoded_depth - depths)): .3f}')
print(f'Hamiltonian: \n\t MAE: {np.mean(np.abs(decoded_depth_ham - depths)): .3f}')

fig, axs = plt.subplots(2, 5, squeeze=False)
axs[0, 0].plot(illums[0, :])
axs[0, 0].set_title('Pulsed Illumination')
axs[0, 0].set_ylabel('Photons')
axs[0, 1].imshow(np.repeat(coding_matrix, 100, axis=0), aspect='auto')
axs[0, 1].set_axis_off()
axs[0, 1].set_title('Coding Matrix (filtered with IRF)')
axs[0, 2].bar(np.arange(n_gates), coded_vals[0, 0, :])
axs[0, 2].set_title('Coded Values')
axs[0, 3].plot(zncc[0, 0, :])
axs[0, 3].axvline(x=depths[0], c='r', linestyle='--')
axs[0, 3].set_title('ZNCC Reconstruction')
axs[0, 4].set_axis_off()

axs[1, 0].plot(modfs[0, :, 0])
axs[1, 0].set_title('Square Illumination')
axs[1, 0].set_ylabel('Photons')

axs[1, 1].imshow(np.repeat(np.transpose(demodfs), 100, axis=0), aspect='auto')
axs[1, 1].set_title('Coding Matrix (Demodulations)')
axs[1, 1].set_axis_off()
axs[1, 2].bar(np.arange(brightness_vals[0, 0, :].shape[0]), brightness_vals[0, 0, :])
axs[1, 2].set_title('Coded Values')
axs[1, 3].plot(zncc_ham[0, 0, :])
axs[1, 3].axvline(x=depths[0], c='r', linestyle='--')
axs[1, 3].set_title('ZNCC Reconstruction')
axs[1, 4].bar([0], np.mean(np.abs(decoded_depth - depths)), label='Coarse')
axs[1, 4].bar([1], np.mean(np.abs(decoded_depth_ham - depths)), label='Hamiltonian')
axs[1, 4].set_title('Mean Error (Lower Better)')
axs[1, 4].legend()
axs[1, 4].set_ylabel('Error')

plt.show()