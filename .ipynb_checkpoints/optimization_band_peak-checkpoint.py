import numpy as np

from felipe_utils import CodingFunctionUtilsFelipe
from utils.file_utils import get_constrained_ham_codes
import cvxpy as cp

(modfs, demodfs) = CodingFunctionUtilsFelipe.GetHamK5(N=1000)
N = modfs.shape[0]
K = modfs.shape[-1]
T = 1  # Total time
dt = T / N  # Time step
threshold = 0.001

window_sizes = [0.15]
peak_powers = [5]
(m, d) = get_constrained_ham_codes(K, peak_powers[0], window_sizes[0], N)

dftmtx = np.fft.fft(np.eye(N))
dftimtx = np.fft.ifft(np.eye(N))

correlation = np.zeros_like(modfs)
for i in range(0, K):
    correlation[:, i] = np.dot(dftimtx, np.dot(dftmtx, modfs[:, i]).conj() * np.dot(dftmtx, demodfs[:, i])).real


for window_size in window_sizes:
    h_t = np.zeros(N)
    h_t[:int(N * window_size)] = np.hanning(N * window_size)
    shift = np.argmax(h_t)
    h_t = np.roll(h_t, shift=-1*shift)
    h_t = h_t / h_t.sum()
    # A = fft(h_t, 2048) / (len(h_t)/2.0)
    # freq = np.linspace(-0.5, 0.5, len(A))
    # response = np.abs(A / abs(A).max())
    # response = 20 * np.log10(np.maximum(response, 1e-10))

    for peak_power in peak_powers:
        constrained_modfs = np.zeros_like(modfs)
        constrained_demodfs = np.zeros_like(demodfs)
        for i in range(0, K):
            corri = correlation[:, i]
            # di = np.ones(N)
            # di[:100] = 0
            # np.random.shuffle(di)
            # di = np.real(np.fft.ifft(np.fft.fft(di) * np.fft.fft(h_t))) / (h_t.sum())
            # di_fft = np.dot(dftmtx, di)
            di_fft = d[:, i]
            flag = True
            prev_optimal = 0
            while True:
                U = cp.Variable(N)
                objective = cp.Minimize(cp.norm(corri -
                                                cp.real(
                                                    dftimtx @ cp.multiply(
                                                        cp.conj(dftmtx @
                                                                cp.real(dftimtx @ cp.multiply(dftmtx @ U, np.fft.fft(h_t)))
                                                                * (1 / h_t.sum())),
                                                        di_fft)))
                                        + 0.5 * cp.tv(cp.real(dftimtx @ cp.multiply(dftmtx @ U, np.fft.fft(h_t)))
                                                      * (1 / h_t.sum())))
                constraints = [
                    cp.sum(cp.multiply(U, dt)) <= 1,
                    U >= 0,
                    U <= peak_power
                ]
                prob = cp.Problem(objective, constraints)
                prob.solve(verbose=True, solver=cp.SCS)

                mi = np.round(U.value)
                mi = np.real(np.fft.ifft(np.fft.fft(mi) * np.fft.fft(h_t))) / (h_t.sum())
                mi_fft = np.dot(dftmtx, mi)
                if flag:
                    mi_fft = m[:, i]
                    flag = False

                optimal_mi = prob.value

                U = cp.Variable(N)
                objective = cp.Minimize(cp.norm(corri -
                                                cp.real(
                                                    dftimtx @ cp.multiply(
                                                        cp.conj(mi_fft),
                                                        dftmtx @ cp.real(dftimtx @ cp.multiply(dftmtx @ U, np.fft.fft(h_t)))
                                                        * (1 / h_t.sum()))))
                                        + 0.5 * cp.tv(cp.real(dftimtx @ cp.multiply(dftmtx @ U, np.fft.fft(h_t)))
                                                      * (1 / h_t.sum())))

                constraints = [
                    U >= 0,
                    U <= 1
                ]
                prob = cp.Problem(objective, constraints)
                prob.solve(verbose=True, solver=cp.SCS)
                di = np.round(U.value)
                di = np.real(np.fft.ifft(np.fft.fft(di) * np.fft.fft(h_t))) / (h_t.sum())
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
        np.save(f'hamk{K}_pmax{peak_power}_wduty{window_size}.npy', full)

print('hello world')
print()

