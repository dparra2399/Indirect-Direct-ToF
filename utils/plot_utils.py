import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

save_folder = '/Users/Patron/Desktop/cowsip figures'
def plot_hist(incident, detected, demodfs):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 10}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

    axs[0].plot(incident)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Intensity')
    axs[0].set_title('Modulation Function')

    axs[1].bar(np.arange(0, detected.shape[0]), detected, edgecolor='black', linewidth=0.1, color='#1f77b4')
    axs[1].set_xlabel('Time bins')
    axs[1].set_ylabel('Photon Count')
    axs[1].set_title('Histogram')

    axs[2].plot(demodfs)
    axs[2].set_xlabel('Time')
    axs[2].set_title('Demodulation Functions')

    fig.savefig(os.path.join(save_folder, 'figure1.jpg'), bbox_inches='tight')
    plt.show()



