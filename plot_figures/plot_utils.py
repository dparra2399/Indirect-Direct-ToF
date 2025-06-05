import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import os
from math import comb
from scipy.stats import binom, poisson


def plot_hist(detected, save_filename):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 10}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots()

    axs.set_xticks([])
    axs.set_yticks([])
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs.bar(np.arange(0, detected.shape[0]), detected, width=2.0, color='blue')
    fig.savefig(save_filename, bbox_inches='tight')
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

def get_scheme_color(coding_scheme, k, cw_tof=False, constant_pulse_energy=False):
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
        if constant_pulse_energy:
            color = 'indigo'
        else:
            color = '#e377c2'
    elif coding_scheme.startswith('KTapSinusoid'):
        if cw_tof:
            color = '#ff7f0eff'
        else:
            color = '#1f77b4'
    elif coding_scheme.startswith('Greys'):
            color = '#d62728'
    elif coding_scheme.startswith('Learned'):
            color = '#1f77b4'
    return color


def darken_cmap(cmap, factor=0.8):
    """Darkens a colormap by a given factor."""
    cdict = cmap._segmentdata
    new_cdict = {}
    for key in ('red', 'green', 'blue'):
        new_cdict[key] = [(x[0], factor * x[1], factor * x[2]) for x in cdict[key]]
    return mcolors.LinearSegmentedColormap('darker_cmap', new_cdict)

