import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from math import comb
from scipy.stats import binom, poisson

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


def plot_modulation_function(modf):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 10}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots()

    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_ylim(0, 30)
    axs.set_title('Modulated Intensity')
    axs.set_xlabel('Time')
    axs.set_ylabel('Intensity')
    axs.plot(modf)
    fig.savefig(os.path.join(save_folder, 'figure1a.jpg'), bbox_inches='tight')
    plt.show()


def plot_modulation_function_with_histogram(modf, hist):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 10}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots()

    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_ylim(0, 30)
    axs.set_title('Measured Histogram')
    axs.set_xlabel('Time Bins')
    axs.set_ylabel('Photon Count')
    axs.plot(modf)
    axs.bar(np.arange(0, hist.shape[0]), hist, alpha=0.5, edgecolor='black', linewidth=1.0, color='#1f77b4')
    fig.savefig(os.path.join(save_folder, 'figure1b.jpg'), bbox_inches='tight')
    plt.show()


def plot_demodulation_functions(demodfs):
    font = {'family': 'normal',
            'size': 10}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots()

    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_title('Demodulation Functions')
    axs.set_xlabel('Time')
    axs.plot(demodfs[:, 0], color='red')
    axs.plot(demodfs[:, 1], color='green')
    axs.plot(demodfs[:, 2], color='purple')
    fig.savefig(os.path.join(save_folder, 'figure1c.jpg'), bbox_inches='tight')
    plt.show()



def calculate_bin_prob(D, theta_bkg, theta_max, total_cycles):
    p_bkg = 1 - np.exp(-theta_bkg/total_cycles)
    p_max = 1 - np.exp(-theta_max/total_cycles)
    expexcted_max = int(D * p_max)
    expexcted_bkg = int(D * p_bkg)
    not_prob = binom.cdf(expexcted_max+expexcted_bkg, D, p_bkg)
    prob = 1-(not_prob)
    print(f'Binomial Percent {prob * 100: .9f} %')

    return prob

def calculate_poisson_prob(theta_bkg, theta_max):
    not_prob_poison = poisson.cdf(theta_max+theta_bkg, theta_bkg)
    prob_poisson = 1-(not_prob_poison**500)
    return prob_poisson