from IPython.core import debugger

from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils import tof_utils_felipe
from utils.file_utils import get_string_name
from plot_figures.plot_utils import *
import matplotlib.pyplot as plt


font = {'family': 'serif',
        'size': 7}

matplotlib.rc('font', **font)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    breakpoint = debugger.set_trace

    save_folder = 'Z:\\Research_Users\\David\\paper figures'
    file = np.load('../data/October', allow_pickle=True)

    params = file['params'].item()
    imaging_schemes_tmp = params['imaging_schemes']

    imaging_schemes = []
    for i in range(len(imaging_schemes_tmp)):

        if imaging_schemes_tmp[i].coding_id.startswith('TruncatedFourier'):
            str_name = 'Truncated Fourier (Wide)' + f'K={imaging_schemes_tmp[i].n_codes}'
            if imaging_schemes_tmp[i].n_codes == 6:
                continue
        elif imaging_schemes_tmp[i].coding_id.startswith('Gated'):
            str_name = 'Coarse Hist. (Wide)' + f'K={imaging_schemes_tmp[i].n_gates}'
            if imaging_schemes_tmp[i].pulse_width == 1:
                continue
        elif imaging_schemes_tmp[i].coding_id.startswith('Hamiltonian'):
            if imaging_schemes_tmp[i].coding_id.endswith('3') or imaging_schemes_tmp[i].coding_id.endswith('5'):
                continue
            str_name = f'SiP Hamiltonian K={imaging_schemes_tmp[i].coding_id[-1]}'
        elif imaging_schemes_tmp[i].coding_id == 'Identity':
            if imaging_schemes_tmp[i].pulse_width == 1:
                str_name = 'Full-Res. Hist. (Narrow)'
            else:
                str_name = 'Full-Res. Hist. (Wide)'
            continue
        elif imaging_schemes_tmp[i].coding_id.startswith('KTapSin'):
            if imaging_schemes_tmp[i].cw_tof is True:
                str_name = 'i-ToF Sinusoid'
            else:
                str_name = 'CoWSiP-ToF Sinusoid'

        elif imaging_schemes_tmp[i].coding_id == 'Greys':
            str_name = 'Count. Greys'
            if imaging_schemes_tmp[i].n_bits != 5:
                continue
            continue
        imaging_schemes.append(imaging_schemes_tmp[i])
    fig, axs = plt.subplots(nrows=len(imaging_schemes), ncols=4)

    axs[0][1].set_title('Incident Waveform')
    axs[0][2].set_title('Demodulation Functions')
    axs[0][3].set_title('Coding Matrix')
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
        axs[i][2].spines['top'].set_visible(False)
        axs[i][2].spines['right'].set_visible(False)
        axs[i][3].spines['top'].set_visible(False)
        axs[i][3].spines['right'].set_visible(False)

        axs[i][1].set_ylabel('Intensity')
        axs[i][1].set_xlabel('Time')
        axs[i][2].set_xlabel('Time')

        if coding_scheme in ['TruncatedFourier', 'Identity', 'Gated', 'Greys']:
            axs[i][2].plot(coding_obj.correlations)
            axs[i][3].imshow(coding_obj.correlations.transpose(), aspect='auto', cmap=plt.cm.get_cmap('binary').reversed())

        else:
            axs[i][2].plot(coding_obj.demodfs)
            axs[i][3].imshow(coding_obj.demodfs.transpose(), aspect='auto', cmap=plt.cm.get_cmap('binary').reversed())


        axs[i][1].plot(np.roll(light_obj.light_source, int(params['n_tbins']/2)), color='blue')
        axs[i][0].set_axis_off()
        axs[i][0].text(0.0, 0.5, f'{get_string_name(imaging_scheme)}')
    #fig.text(0.04, 0.25, 'Hamiltonian', va='center', rotation='vertical', fontsize=7)

    fig.suptitle('Coding Scheme Tested', fontsize=12,  fontweight="bold")
    fig.tight_layout()  # Adjust the rect to make space for the common labels


    for i in range(len(axs) - 1):
        row_bottom = axs[i, 0].get_position().y0

        # Add horizontal line
        fig.add_artist(plt.Line2D(
            [0, 1], [row_bottom-0.035, row_bottom-0.035], transform=fig.transFigure,
            color='black', linewidth=1
        ))

    row_top = axs[0, 0].get_position().y1

    # Add horizontal line
    fig.add_artist(plt.Line2D(
        [0, 1], [row_top+0.05, row_top+0.05], transform=fig.transFigure,
        color='black', linewidth=1
    ))

    for j in range(len(axs[0])):
        col_left = axs[0, j].get_position().x1
        if j > 0:
            col_left += 0.02
        else:
            col_left -= 0.02
        fig.add_artist(plt.Line2D(
            [col_left, col_left], [0, 0.935], transform=fig.transFigure,
            color='black', linewidth=1
        ))

    fig.savefig('Z:\\Research_Users\\David\\paper figures\\supfigure1.svg', bbox_inches='tight')
    plt.show()

