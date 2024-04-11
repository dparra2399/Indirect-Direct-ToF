import numpy as np

from felipe_utils.felipe_cw_utils import CodingFunctionUtilsFelipe
from spad_toflib.spad_tof_utils import gaussian_irf
import cvxpy as cp
import matplotlib.pyplot as plt

(modfs, demodfs) = CodingFunctionUtilsFelipe.GetHamK3(N=1000)
A = 1.0  # Amplitude
sigma = 1.0  # Standard deviation
t = np.linspace(-5, 5, 1000)  # Time range

# Compute the IRF
h_t = A * np.exp(-0.5 * (t / sigma)**2)
one = np.convolve(modfs[:, 0], h_t, mode='same')

N = modfs.shape[0]
K = modfs.shape[-1]
T = 1  # Total time
dt = T / N  # Time step

dftmtx = np.fft.fft(np.eye(N))
dftimtx = np.fft.ifft(np.eye(N))

correlation = np.zeros_like(modfs)
for i in range(0, K):
    correlation[:, i] = np.dot(dftimtx, np.dot(dftmtx, modfs[:, i]).conj() * np.dot(dftmtx, demodfs[:, i])).real


constrained_modfs = np.zeros_like(modfs)
for i in range(0, K):
    U = cp.Variable(N, integer=True)
    corri = correlation[:, i]
    di = demodfs[:, i]
    mi = modfs[:, i]
    objective = cp.Minimize(cp.norm(corri -
                            cp.real(dftimtx @ cp.multiply(cp.conj(dftmtx @ U), dftmtx @ di))))
    constraints = [
        cp.sum(cp.multiply(U, dt)) <= 1,
        U >= 0,
        U <= 2
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.SCIP, max_iters=50)
    constrained_modfs[:, i] = U.value
    #plt.plot(corri_corr)

#plt.plot(U.value)
plt.plot(constrained_modfs)
plt.show()

print('hello world')
print()