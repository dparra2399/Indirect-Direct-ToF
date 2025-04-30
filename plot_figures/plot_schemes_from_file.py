from IPython.core import debugger

from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from utils.file_utils import get_string_name
from plot_figures.plot_utils import *
import matplotlib.pyplot as plt
from felipe_utils.research_utils.signalproc_ops import gaussian_pulse, smooth_codes
import matplotlib.patches as patches


font = {'family': 'serif',
        'size': 7}

matplotlib.rc('font', **font)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    breakpoint = debugger.set_trace

    file = np.load('../data/results/OLD STUFF/ntbins_1024_monte_1000_exp_Learned005.npz',allow_pickle=True)

    params = file['params'].item()
    imaging_schemes = params['imaging_schemes']

    fig, axs = plt.subplots(nrows=len(imaging_schemes), ncols=4, figsize=(8, 4))

    # np.save(
    #     r'C:\Users\Patron\PycharmProjects\Indirect-Direct-ToF\learned_codes\bandlimited_models\n1024_k4_sigma30\coded_model.npy',
    #     imaging_schemes[2].coding_obj.correlations)
    #
    # np.save(
    #     r'C:\Users\Patron\PycharmProjects\Indirect-Direct-ToF\learned_codes\bandlimited_models\n1024_k4_sigma30\illum_model.npy',
    #     imaging_schemes[2].light_obj.light_source[0, :])




    axs[0][1].set_title('Input Illumination f(t)')
    axs[0][3].set_title('Rows of D')
    axs[0][2].set_title('Coding Matrix D')
    for i in range(len(imaging_schemes)):
        imaging_scheme = imaging_schemes[i]
        coding_obj = imaging_scheme.coding_obj
        coding_scheme = imaging_scheme.coding_id
        light_obj = imaging_scheme.light_obj
        light_source = imaging_scheme.light_id
        rec_algo = imaging_scheme.rec_algo

        axs[i][1].set_xticks([])
        axs[i][1].set_yticks([])
        axs[i][2].set_xticks([])
        axs[i][2].set_yticks([])
        axs[i][3].set_xticks([])
        axs[i][3].set_yticks([])
        axs[i][1].spines['top'].set_visible(False)
        axs[i][1].spines['right'].set_visible(False)

        axs[i][1].set_ylabel('Intensity')
        axs[i][1].set_xlabel('Time')
        axs[i][3].set_xlabel('Time')

        axs[i][3].plot(coding_obj.decode_corrfs)
        #axs[i][2].set_ylim(-1, 1)
        tmp = coding_obj.decode_corrfs.transpose()
        #tmp[4, :] = 0

        img = np.repeat(coding_obj.decode_corrfs.transpose(), 100, axis=0)
        axs[i][2].imshow(img, cmap='gray', aspect='auto')

        axs[i][2].set_axis_off()
        irf = gaussian_pulse(np.arange(params['n_tbins']), 0, 30, circ_shifted=True)

        first_zero_index = np.where(light_obj.light_source == 0)[0]
        axs[i][1].plot(np.roll(irf, 512), color='blue')
        axs[i][1].set_xticks([])
        axs[i][0].set_axis_off()
        axs[i][0].text(0.0, 0.5, f'{get_string_name(imaging_scheme)}')

    fig.suptitle(f'Coding Scheme with IRF width $\sigma$={30}$\Delta$', fontsize=12,  fontweight="bold")
    fig.tight_layout()  # Adjust the rect to make space for the common labels



    for i in range(len(axs) - 1):
        row_bottom = axs[i, 0].get_position().y0

        # Add horizontal line
        fig.add_artist(plt.Line2D(
            [0, 1], [row_bottom-0.04, row_bottom-0.04], transform=fig.transFigure,
            color='black', linewidth=1
        ))

    row_top = axs[0, 0].get_position().y1

    # Add horizontal line
    fig.add_artist(plt.Line2D(
        [0, 1], [row_top+0.08, row_top+0.08], transform=fig.transFigure,
        color='black', linewidth=1
    ))

    for j in range(len(axs[0]) - 1):
        col_left = axs[0, j].get_position().x1
        if j > 0:
            col_left += 0.02
        else:
            col_left -= 0.02
        fig.add_artist(plt.Line2D(
            [col_left, col_left], [0, 0.935], transform=fig.transFigure,
            color='black', linewidth=1
        ))


    fig.savefig(f'Z:\\Research_Users\\David\\Learned Coding Functions Paper\\supp_k4.svg', bbox_inches='tight', dpi=300)
    plt.show()