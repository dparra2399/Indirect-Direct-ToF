import numpy as np
import matplotlib.pyplot as plt
from felipe_utils import tof_utils_felipe

rep_tau = 1/(1.5 * 1e6)
d_max = 100
bit_size = 8
hist_sizes = np.linspace(1000, 200_000, num=1000)
depth_res = tof_utils_felipe.time2depth(rep_tau / hist_sizes) * 1000
fps = 30
size = 240*160

full_res_gb = hist_sizes * fps * size * bit_size * 1e-9
ham_k4_gb = (4 * fps + (4 * hist_sizes* bit_size)) * 1e-9
coarse_gb = hist_sizes / 10 * fps * size * bit_size * 1e-9

plt.plot(full_res_gb, label='Full-resolution hist')
plt.plot(ham_k4_gb, label='Hamiltonian K=4 (Our Method)')
plt.plot(coarse_gb, label='Coarse and Compressive Hist. (10X reduction)')
plt.xticks(np.arange(0, 1000, step=100), np.round(depth_res[::100], decimals=2))
plt.xlabel('Depth Resolution (mm)')
plt.ylabel('GB per Second')
plt.title(f'Data rates at {fps}fps')
plt.legend()
plt.show()
