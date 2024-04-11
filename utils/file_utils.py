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
