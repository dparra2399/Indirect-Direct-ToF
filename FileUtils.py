# Python imports
# Library imports
import numpy as np
from IPython.core import debugger

breakpoint = debugger.set_trace

import os.path


def WriteErrorsToFile(params, results, exp_num, depths, sbr_levels, pAveAmbient_levels, pAveSource_levels):
    n_tbins = params['n_tbins']
    trials = params['trials']
    lvls = results['mae_itof'].shape[0]


    filename = 'ntbins_{}_monte_{}_lvls_{}_exp_{}.npz'.format(n_tbins, trials, lvls, exp_num)
    outfile = './data/results/' + filename

    np.savez(outfile, params=params, results=results, exp_num=exp_num, depths=depths,
             sbr_levels=sbr_levels, pAveAmbient_levels=pAveAmbient_levels, pAveSource_levels=pAveSource_levels)

    if os.path.isfile(outfile):
        print("Filename {} overwritten".format(filename))
