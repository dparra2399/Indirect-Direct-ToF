# Python imports
# Library imports
import numpy as np
from IPython.core import debugger

breakpoint = debugger.set_trace

import os.path


def WriteErrorsToFile(Experiment,trials, depths, dMax,
            n_tbins, pAveAmbientList, pAveSourceList, freq, tau,
            depth_padding, tbin_res, tbin_depth_res, t_domain,
            gt_tshifts, rec_algo, pw_factors, K, T, dt, meanBeta,
            mae_idtof, mae_itof, mae_dtof):

    filename = 'ntbins_{}_monte_{}_exp_{}.npz'.format(n_tbins, trials,  Experiment)
    outfile = './data/results/' + filename


    np.savez(outfile, trials=trials, depths=depths, dMax=dMax, n_tbins=n_tbins,
            pAveAmbientList=pAveAmbientList,pAveSourceList=pAveSourceList, freq=freq, tau=tau,
            depth_padding=depth_padding, tbin_res=tbin_res,
            tbin_depth_res=tbin_depth_res, t_domain=t_domain,
            gt_tshifts=gt_tshifts, rec_algo=rec_algo, pw_factors=pw_factors,
            K=K, T=T, dt=dt, meanBeta=meanBeta, mae_idtof=mae_idtof,
             mae_itof=mae_itof, mae_dtof=mae_dtof)

    if os.path.isfile(outfile):
        print("Filename {} overwritten".format(filename))
