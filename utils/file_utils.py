# Python imports
# Library imports
import numpy as np
from IPython.core import debugger

breakpoint = debugger.set_trace

import os.path


def write_errors_to_file(params, results, depths, levels_one, levels_two):
    n_tbins = params['n_tbins']
    trials = params['trials']

    filename = 'ntbins_{}_monte_{}.npz'.format(n_tbins, trials)
    outfile = './data/results/' + filename

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
