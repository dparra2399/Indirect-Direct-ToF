# Python imports
# Library imports
import numpy as np
from IPython.core import debugger
import matplotlib.pyplot as plt

breakpoint = debugger.set_trace

import os.path


def write_errors_to_file(params, results, depths, levels_one, levels_two, exp):
    n_tbins = params['n_tbins']
    trials = params['trials']

    filename = 'ntbins_{}_monte_{}_exp_{}.npz'.format(n_tbins, trials, exp)
    outfile = '.\\data\\results\\' + filename

    np.savez(outfile, params=params, results=results, depths=depths,
             levels_one=levels_one, levels_two=levels_two)

    if os.path.isfile(outfile):
        print("Filename {} overwritten".format(filename))


def get_constrained_ham_codes(k, peak_power, win_duty, n_tbins):
    filename = f"hamk{k}_pmax{peak_power}_wduty{win_duty}.npy"
    dir = './optimizations/' + filename
    try:
        file = np.load(dir)
    except FileNotFoundError:
        assert False, f'Constrained Ham Codes not Implemented for {peak_power} peak power and {win_duty} window IRF'
    (modfs, demodfs) = (file[0, ...], file[1, ...])
    assert n_tbins <= modfs.shape[0], f'Constrained Ham Codes not Implemented for {n_tbins} time bins'
    assert modfs.shape[0] % n_tbins == 0, f'Make Time bins a multiple of {modfs.shape[0]}'
    subsample = int(modfs.shape[0] // n_tbins)
    return modfs[::subsample], demodfs[::subsample]

def get_string_name(imaging_scheme):
    if imaging_scheme.coding_id == 'Identity':
        assert imaging_scheme.light_id == 'Gaussian'
        PW = imaging_scheme.pulse_width
        if PW > 1:
            str_name = r'$\bf{FRH}$' + r' $\bf{(Wide)}$'
        else:
            str_name = r'$\bf{FRH}$' + r' $\bf{(Narrow)}$'
        str_name = r'Full Res. Histogram' + f'\n $\sigma = {PW}\Delta$'
        #str_name += 'Full-resolution Hist.'
    elif imaging_scheme.coding_id == 'Gated':
        assert imaging_scheme.light_id == 'Gaussian'
        PW = imaging_scheme.pulse_width
        n_gates = imaging_scheme.n_gates
        if PW > 1:
            str_name = r'Coarse' + r' $\bf{(Wide)}$'
        else:
            str_name = r'Coarse' + r' $\bf{(Narrow)}$'
        str_name = r'$\bf{Coarse} \bf{Hist.}$ ' + f'\n $K={n_gates}, \sigma = {PW}\Delta$'
    elif imaging_scheme.coding_id[:-2] == 'Hamiltonian':
        if imaging_scheme.account_irf is True:
            str_name = f'Ham{imaging_scheme.coding_id[-2:]} IRF'
        else:
            #str_name += f'Ham{imaging_scheme.coding_id[-2:]} D = {duty}%'
            str_name = r'SiP Hamiltonian' + f'\n K={imaging_scheme.coding_id[-1:]}'
    elif imaging_scheme.coding_id == 'KTapSinusoid':
        ktaps = imaging_scheme.ktaps
        cw_tof = imaging_scheme.cw_tof
        if cw_tof:
            str_name = f'CW Sinusoid K={ktaps}'
        else:
            str_name = f'SP Sinusoid K={ktaps}'
    elif imaging_scheme.coding_id == 'TruncatedFourier':
        k = (imaging_scheme.n_freqs + 1) * 2
        PW = imaging_scheme.pulse_width
        if PW > 1:
            str_name = r'$\bf{Trunc. Fourier}$' + r' $\bf{(Wide)}$'
        else:
            str_name = r'$\bf{Trunc. Fourier}$' + r' $\bf{(Narrow)}$'
        str_name = r'Truncated Fourier' + f'\n K={k}, $\sigma = {PW}\Delta$'
    elif imaging_scheme.coding_id == 'Greys':
        k = imaging_scheme.n_bits
        PW = imaging_scheme.pulse_width
        str_name = r'Continuous Greys' + f'\n K={k}, $\sigma = {PW}\Delta$'
    else:
        str_name = f'{imaging_scheme.coding_id}'
    return str_name