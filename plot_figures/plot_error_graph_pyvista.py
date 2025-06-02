import numpy as np
import pyvista as pv
from felipe_utils import tof_utils_felipe
from utils.file_utils import get_string_name
from plot_figures.plot_utils import get_scheme_color



if __name__ == '__main__':
    # File setup
    filenames = [
        '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma10_peak015_mae.npz',
        '../data/results/bandlimit_peak_simulation/ntbins_1024_monte_5000_exp_Learned_sigma10_peak015_rmse.npz',
    ]

    num = 8 # high SBR
    num2 = 2  # Low Photon count
    num3 = 1  # Low SBR
    num4 = 1  # High photon count

    # PyVista plotter
    plotter = pv.Plotter(shape=(1, len(filenames)), window_size=(1400, 600))

    for i, filename in enumerate(filenames):
        file = np.load(filename, allow_pickle=True)
        mae = file['results'][:, num3:-num, num2:-num4] * (1/10)
        levels_one = file['levels_one'][num3:-num, num2:-num4]
        levels_two = file['levels_two'][num3:-num, num2:-num4]
        params = file['params'].item()
        imaging_schemes = params['imaging_schemes']

        X, Y = np.meshgrid(levels_one, levels_two, indexing='ij')

        plotter.subplot(0, i)
        plotter.set_background("white")

        for j in range(len(imaging_schemes)):
            tmp = mae[j, :, :]
            coding = imaging_schemes[j].coding_id

            # Filter some schemes (mimicking original logic)
            if coding.startswith('TruncatedFourier') and imaging_schemes[j].n_codes == 6:
                continue
            if coding.startswith('Hamiltonian') and coding.endswith(('3', '5')):
                continue

            k = 4  # fixed since coding_obj.n_functions isn't stored
            label = get_string_name(imaging_schemes[j])
            color = get_scheme_color(coding, k, cw_tof=imaging_schemes[j].cw_tof)

            # Create PyVista mesh
            levels_one_norm = (np.log10(levels_one) - np.log10(levels_one).min()) / (np.log10(levels_one).max() - np.log10(levels_one).min())
            levels_two_norm = (np.log10(levels_two)  - np.log10(levels_two).min()) / (np.log10(levels_two).max() - np.log10(levels_two).min())
            tmp_norm = (tmp - mae.min()) / (mae.max() - mae.min())

            grid = pv.StructuredGrid(levels_one_norm, levels_two_norm, tmp_norm)
            plotter.add_mesh(grid, name=label,
                             color=color,
                             show_edges=True)
            plotter.show_bounds(location='origin')

        plotter.add_axes()
        #plotter.add_text("RMSE" if 'rmse' in filename else "MAE", font_size=12)
        zlim = (0, 60) if 'peak015' in filename else (0, 50)
        # plotter.show_bounds(axes_ranges=[np.min(levels_one), np.max(levels_one), np.min(levels_two), np.max(levels_two),
        #                                  zlim[0], zlim[1]])
        # plotter.set_xlabel("Log Photon Count")
        # plotter.set_ylabel("Log SBR")
        # plotter.set_zlabel("Error (cm)")

    plotter.link_views()
    plotter.view_vector([10, -30, 10], [0, 0, 1])

    plotter.show()
