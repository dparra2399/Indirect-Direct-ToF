import numpy as np

from felipe_utils.felipe_cw_utils import CodingFunctionUtilsFelipe
from spad_toflib.spad_tof_utils import gaussian_irf
from scipy.fft import fft, fftshift
from utils.file_utils import get_constrained_ham_codes
import cvxpy as cp
import matplotlib.pyplot as plt

(modfs, demodfs) = CodingFunctionUtilsFelipe.GetHamK5(N=1000)
N = modfs.shape[0]
K = modfs.shape[-1]
T = 1  # Total time
dt = T / N  # Time step
threshold = 0.001

peak_powers = [5]

dftmtx = np.fft.fft(np.eye(N))
dftimtx = np.fft.ifft(np.eye(N))

correlation = np.zeros_like(modfs)
for i in range(0, K):
    correlation[:, i] = np.dot(dftimtx, np.dot(dftmtx, modfs[:, i]).conj() * np.dot(dftmtx, demodfs[:, i])).real


for peak_power in peak_powers:
    constrained_modfs = np.zeros_like(modfs)
    constrained_demodfs = np.zeros_like(demodfs)
    for i in range(0, K):
        corri = correlation[:, i]
        di = np.ones(N)
        di[:100] = 0
        np.random.shuffle(di)
        di_fft = np.dot(dftmtx, di)
        flag = True
        prev_optimal = 0
        while True:
            U = cp.Variable(N)
            objective = cp.Minimize(cp.norm(corri -
                                            cp.real(dftimtx @ cp.multiply( cp.conj(dftmtx @ U), di_fft)))
                                    + 1.0 * cp.tv(U))
            constraints = [
                cp.sum(cp.multiply(U, dt)) <= 1,
                U >= 0,
                U <= peak_power
            ]
            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=True, solver=cp.SCS)

            mi = np.round(U.value)
            mi_fft = np.dot(dftmtx, mi)

            optimal_mi = prob.value

            U = cp.Variable(N)
            objective = cp.Minimize(cp.norm(corri -
                                            cp.real(dftimtx @ cp.multiply(cp.conj(mi_fft), dftmtx @ U)))
                                    + 1.0 * cp.tv(U))

            constraints = [
                U >= 0,
                U <= 1
            ]
            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=True, solver=cp.SCS)
            di = np.round(U.value)
            di_fft = np.dot(dftmtx, di)
            if np.abs(prev_optimal - optimal_mi) <= threshold:
                print("The optimal value is", prob.value)
                break
            prev_optimal = optimal_mi
        constrained_modfs[:, i] = mi
        constrained_demodfs[:, i] = di

    # plt.plot(U.value)
    #plt.plot(constrained_modfs)
    #plt.plot(constrained_demodfs)
    #plt.show()
    full = np.stack((constrained_modfs, constrained_demodfs), axis=0)
    np.save(f'hamk{K}_pmax{peak_power}.npy', full)

print('hello world')
print()
