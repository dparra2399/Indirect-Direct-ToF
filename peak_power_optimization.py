import numpy as np

from felipe_utils.felipe_cw_utils import CodingFunctionUtilsFelipe
from spad_toflib.spad_tof_utils import gaussian_irf
import cvxpy as cp
import matplotlib.pyplot as plt

(modfs, demodfs) = CodingFunctionUtilsFelipe.GetHamK4(N=1000)

N = modfs.shape[0]
K = modfs.shape[-1]
T = 1  # Total time
dt = T / N  # Time step
threshold = 0.5

dftmtx = np.fft.fft(np.eye(N))
dftimtx = np.fft.ifft(np.eye(N))

correlation = np.zeros_like(modfs)
for i in range(0, K):
    correlation[:, i] = np.dot(dftimtx, np.dot(dftmtx, modfs[:, i]).conj() * np.dot(dftmtx, demodfs[:, i])).real


constrained_modfs = np.zeros_like(modfs)
constrained_demodfs = np.zeros_like(demodfs)

for i in range(0, K):
    corri = correlation[:, i]
    di = np.ones(N)
    di[:100] = 0
    np.random.shuffle(di)
    prev_optimal = 0
    while True:
        U = cp.Variable(N)
        objective = cp.Minimize(cp.norm(corri -
                                cp.real(dftimtx @ cp.multiply(cp.conj(dftmtx @ U), dftmtx @ di)))
                                + 0.5 * cp.tv(U))
        constraints = [
            cp.sum(cp.multiply(U, dt)) <= 1,
            U >= 0,
            U <= 2
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=True, solver=cp.SCS)

        mi = np.round(U.value)
        optimal_mi = prob.value

        U = cp.Variable(N)
        objective = cp.Minimize(cp.norm(corri -
                                cp.real(dftimtx @ cp.multiply(cp.conj(dftmtx @ mi), dftmtx @ U)))
                                + 0.5 * cp.tv(U))

        constraints = [
            U >= 0,
            U <= 1
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=True, solver=cp.SCS)
        di = np.round(U.value)
        if np.abs(prev_optimal - optimal_mi) <= threshold:
            print("The optimal value is", prob.value)
            break
        prev_optimal = optimal_mi
    constrained_modfs[:, i] = mi
    constrained_demodfs[:, i] = di

#plt.plot(U.value)
plt.plot(constrained_modfs)
plt.plot(constrained_demodfs)
plt.show()
#np.save('hamk4-pp2.npy', constrained_modfs)
print('hello world')
print()

bandwidth = 1 * 1e6
t = np.linspace(-5, 5, 1000)
sigma = bandwidth * np.sqrt(2 * np.log(2))
impulse_response = 1 / np.sqrt(4 * np.pi * (bandwidth**2) * np.log(2)) * np.exp(-t**2 / (4 * (bandwidth**2) * np.log(2)))


one = np.convolve(modfs[:, 0], impulse_response, mode='same')
