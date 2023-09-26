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
from combined_indirect_tof import combined_indirect_utils, CodingFunctions
breakpoint = debugger.set_trace
from direct_toflib import tirf
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
    meanBeta = 1  # Avg fraction of photons reflected from a scene points back to the detector

    ##DIRECT
    depth_padding = 0.02  # Skip the depths at the boundaries
    pw_factors = np.array([0.5])
    rec_algo = 'linear'

    (rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
        (direct_tof_utils.calc_tof_domain_params(n_tbins, rep_tau=rep_tau))


    sourceExponent = 9
    ambientExponent = 9

    pAveSourcePerPixel = np.power(10, sourceExponent)
    pAveAmbientPerPixel = np.power(10, ambientExponent)


    depths = np.array([5.32, 6.78, 2.01, 7.68, 8.34])
    gt_tshifts = direct_tof_utils.depth2time(depths)

    (ModFs, DemodFs) = CodingFunctions.GetCosCos(N=n_tbins, K=K)

    kappas = np.sum(DemodFs, 0) * dt
    Ambient = rep_tau * pAveSourcePerPixel * kappas

    #ModFs = combined_indirect_utils.ScaleMod(ModFs, tau=rep_tau, pAveSource=pAveSourcePerPixel)
    Incident = (rep_tau * pAveSourcePerPixel) * meanBeta * (ModFs + Ambient)

    pulses_list = tirf.init_gauss_pulse_list(n_tbins, pw_factors * tbin_res, mu=gt_tshifts, t_domain=t_domain)
    pulses = pulses_list[0]

    curr_sbr = pAveSourcePerPixel / pAveAmbientPerPixel
    n_photons = rep_tau * pAveSourcePerPixel #total photons
    pulses.set_sbr(curr_sbr)

    simulated_pulses = pulses.simulate_n_signal_photons(n_photons=n_photons, n_mc_samples=1)

    plot_pulses = np.transpose(np.squeeze(simulated_pulses))
    plot_clean_puleses = np.transpose(np.squeeze(pulses.tirf))
    plt.plot(plot_pulses)
    #plt.plot(plot_clean_puleses)
    plt.plot(Incident)
    plt.show()
    print("hello world")