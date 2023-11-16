# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core import debugger
breakpoint = debugger.set_trace
mpl.use('qt5agg')

import simulate_tof_scene
from direct_toflib import direct_tof_utils
from indirect_toflib import indirect_tof_utils, CodingFunctions
breakpoint = debugger.set_trace
from direct_toflib import tirf
import debug_utils
import FileUtils


if __name__ == "__main__":
    n_tbins = 1024
    K = 4
    T = 0.1  # Integration time. Exposure time in seconds
    speedOfLight = 299792458. * 1000.  # mm / sec
    rep_freq = 1e+7
    rep_tau = 1. / rep_freq
    dMax = direct_tof_utils.time2depth(rep_tau)
    fSampling = float(dMax) * rep_freq  # Sampling frequency of mod and demod functuion
    dt = rep_tau / float(n_tbins)
    meanBeta = 1e-4  # Avg fraction of photons reflected from a scene points back to the detector

    ##DIRECT
    rec_algo = 'linear'

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (direct_tof_utils.calc_tof_domain_params(n_tbins, rep_tau=rep_tau))

    pAveSource = 10**9
    sbr = 10**-1

    ############### INDIRECT ####################
    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)
    ModFs = indirect_tof_utils.ScaleMod(ModFs, tau=rep_tau, pAveSource=pAveSource)
    peak_power = np.max(ModFs)


    ################### DIRECT ###################
    depths = np.array([5.5, 6.5, 4.3])
    gt_tshifts = direct_tof_utils.depth2time(depths)

    pw_factors = np.array([1])
    sigma = pw_factors * tbin_res

    pulses_list = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=gt_tshifts, t_domain=t_domain)

    pulses = pulses_list[0]

    pulse = np.squeeze(pulses.tirf)
    # plt.plot(pulse)

    pulses.set_sbr(None)
    simulated_pulses = pulses.simulate_n_signal_photons(n_photons=pAveSource,
                                                        n_mc_samples=1,
                                                        peak_power=peak_power,
                                                        num_mes=K,
                                                        add_noise=False)

    plot_pulses = np.transpose(np.squeeze(simulated_pulses))
    plt.plot(plot_pulses)
    plt.plot(ModFs)

    plt.show()





