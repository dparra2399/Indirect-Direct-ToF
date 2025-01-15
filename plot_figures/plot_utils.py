import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import os
from math import comb
from scipy.stats import binom, poisson

save_folder = 'Z:\\Research_Users\\David\\paper figures'


def plot_hist(detected):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 10}

    matplotlib.rc('font', **font)

    save_folder = 'Z:\\Research_Users\\David\\paper figures'
    fig, axs = plt.subplots()

    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs.bar(np.arange(0, detected.shape[0]), detected, edgecolor='black', linewidth=0.1, color='blue')
    fig.savefig(os.path.join(save_folder, 'figure1b.svg'), bbox_inches='tight')
    plt.show()


def plot_modulation_function(modf, pulsed=False, filename=None):

    fig, axs = plt.subplots()

    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.plot(modf, color='blue')
    if pulsed is not False:
        axs.plot(pulsed, color='red')
    fig.savefig(os.path.join(save_folder, f'{filename}.svg'), bbox_inches='tight')
    plt.show()


def plot_modulation_function_with_histogram(modf, hist, filename=None):
    font = {'family': 'serif',
            'size': 10}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots()

    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    #axs.set_ylim(0, 30)
    #axs.set_title('Measured Histogram')
    #axs.set_xlabel('Time Bins')
    #axs.set_ylabel('Photon Count')
    axs.bar(np.arange(0, hist.shape[0]), hist, edgecolor='black', linewidth=0.5, color='blue')
    fig.savefig(os.path.join(save_folder, f'whatever.svg'), bbox_inches='tight')
    plt.show()


def plot_demodulation_functions(demodfs, filename=None):
    font = {'family': 'serif',
            'size': 10}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots()

    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    #axs.set_title('Demodulation Functions')
    #axs.set_xlabel('Time')
    axs.plot(demodfs[:, 0], color='salmon')
    axs.plot(demodfs[:, 1], color='limegreen')
    axs.plot(demodfs[:, 2], color='violet')
    fig.savefig(os.path.join(save_folder, f'{filename}.svg'), bbox_inches='tight')
    plt.show()

def plot_correlation_functions(correlation, b_vals, d_hat, filename=None):
    font = {'family': 'serif',
            'size': 10}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots()

    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.set_ylim(-2, 2)
    axs.plot(correlation[:, 0], color='salmon')
    axs.plot(correlation[:, 1], color='limegreen')
    axs.plot(correlation[:, 2], color='violet')
    axs.scatter(x=d_hat, y=b_vals[0], color='blue')
    axs.scatter(x=d_hat, y=b_vals[1], color='blue')
    axs.scatter(x=d_hat, y=b_vals[2], color='blue')
    axs.axvline(x=d_hat, color='orange')
    fig.savefig(os.path.join(save_folder, f'{filename}.svg'), bbox_inches='tight')
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

def get_scheme_color(coding_scheme, k, cw_tof=False):
    color = None
    if coding_scheme.startswith('TruncatedFourier'):
        if k==6:
            color = 'gold'
        else:
            color = '#ff7f0eff'
    elif coding_scheme.startswith('Gated'):
        color = '#2ca02c'
    elif coding_scheme.startswith('Hamiltonian'):
        if k==5:
            color = '#1f77b4'
        elif k==4:
            color = '#1f77b4'
    elif coding_scheme == 'Identity':
        color = '#e377c2'
    elif coding_scheme.startswith('KTapSinusoid'):
        if cw_tof:
            color = '#ff7f0eff'
        else:
            color = '#1f77b4'
    elif coding_scheme.startswith('Greys'):
            color = '#d62728'
    return color


def darken_cmap(cmap, factor=0.8):
    """Darkens a colormap by a given factor."""
    cdict = cmap._segmentdata
    new_cdict = {}
    for key in ('red', 'green', 'blue'):
        new_cdict[key] = [(x[0], factor * x[1], factor * x[2]) for x in cdict[key]]
    return mcolors.LinearSegmentedColormap('darker_cmap', new_cdict)

