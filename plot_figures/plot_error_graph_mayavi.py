from mayavi import mlab
import numpy as np
from felipe_utils import tof_utils_felipe
from utils.file_utils import get_string_name
from plot_figures.plot_utils import get_scheme_color
import matplotlib.colors as mcolors

filenames = [
    '../data/results/bandlimit_simulation/ntbins_1024_monte_5000_exp_Learned_sigma30_rmse.npz',
]

num = 1
num2 = 2
num3 = 1
num4 = 1

for filename in filenames:
    file = np.load(filename, allow_pickle=True)

    mae = file['results'][:, num3:-num, num2:-num4] * (1 / 10)
    levels_one = file['levels_one'][num3:-num, num2:-num4]
    levels_two = file['levels_two'][num3:-num, num2:-num4]
    params = file['params'].item()
    imaging_schemes = params['imaging_schemes']
    tbin_res = params['rep_tau'] / params['n_tbins']
    tbin_depth_res = tof_utils_felipe.time2depth(tbin_res)

    #levels_one = np.log10(levels_one)
    #levels_two = np.log10(levels_two)

    for j in range(len(imaging_schemes)):
        tmp = mae[j, :, :]

        # Skip based on imaging scheme if needed
        coding_id = imaging_schemes[j].coding_id
        if coding_id.startswith('TruncatedFourier') and imaging_schemes[j].n_codes == 6:
            continue
        if coding_id.startswith('Hamiltonian') and coding_id.endswith(('3', '5')):
            continue

        label = get_string_name(imaging_schemes[j])
        k = 4  # Or dynamically: imaging_schemes[j].n_codes or .n_functions
        color = get_scheme_color(coding_id, k,
                                 cw_tof=imaging_schemes[j].cw_tof,
                                 constant_pulse_energy=imaging_schemes[j].constant_pulse_energy)

        X = (np.log10(levels_one) - np.log10(levels_one).min()) / (
                    np.log10(levels_one).max() - np.log10(levels_one).min())
        Y = (np.log10(levels_two) - np.log10(levels_two).min()) / (
                    np.log10(levels_two).max() - np.log10(levels_two).min())
        Z = (tmp - mae.min()) / (mae.max() - mae.min())
        # Plot with Mayavi
        surf = mlab.mesh(X, Y, Z, color=mcolors.to_rgb(color) )
        #mlab.outline()
        a = mlab.axes(s,
                      z_axis_visibility=True,
                      x_axis_visibility=False,
                      y_axis_visibility=False,
                      ranges=[X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()],
                      color=(0, 0, 0)
                      )
    mlab.view(azimuth=-45, elevation=15, distance='auto')

mlab.show()
