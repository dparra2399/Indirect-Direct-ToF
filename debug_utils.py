### Python imports
#### Library imports
import numpy as np
from scipy import stats
from IPython.core import debugger
import matplotlib.pyplot as plt

from indirect_toflib.indirect_tof_utils import AddPoissonNoiseLam
from combined_toflib.combined_tof_utils import AddPoissonNoiseArr
from direct_toflib import direct_tof_utils, tirf
breakpoint = debugger.set_trace


def debbug_cw(params, ModFs, Incident, NormMeasures, norm_measurements_idtof,  norm_measurements_itof,
                depths, sbr, pAveAmbient, pAveSource):


    tbin_depth_res = direct_tof_utils.time2depth(params['rep_tau'] / params['n_tbins'])
    figure, axis = plt.subplots(2, depths.shape[0])
    for j in range(0, depths.shape[0]):
        tmp = np.dot(NormMeasures, norm_measurements_idtof.transpose())
        id = np.squeeze(tmp[:, j])

        tmp2 = np.dot(NormMeasures, norm_measurements_itof.transpose())
        ind = np.squeeze(tmp2[:, j])

        if depths.shape[0] > 1:
            axis[0, j].plot(NormMeasures)
            axis[0, j].axvline(x=depths[j], label='depth')
            axis[0, j].axvline(x=np.argmax(id), color='pink', label='comb measures')
            axis[0, j].axvline(x=np.argmax(ind), color='red', label='indirect measures')
            axis[0, j].set_title('Ground truth measures vs true measures at depth : ' + str(depths[j]*tbin_depth_res))
            axis[0, j].legend()
        else:
            axis[0].plot(NormMeasures)
            axis[0].axvline(x=depths[j], label='depth')
            axis[0].axvline(x=np.argmax(id), color='pink', label='comb measures')
            axis[0].axvline(x=np.argmax(ind), color='red', label='indirect measures')
            axis[0].set_title('Ground truth measures vs true measures at depth : ' + str(depths[j]*tbin_depth_res))
            axis[0].legend()

        for i in range(0, norm_measurements_idtof.shape[1]):
            x = np.argmax(id)
            y = norm_measurements_idtof[j, i]

            if depths.shape[0] > 1:
                axis[0, j].scatter(x, y, color='red', marker='o', label='Point')
            else:
                axis[0].scatter(x, y, color='red', marker='o', label='Point')

            x = np.argmax(ind)
            y = norm_measurements_itof[j, i]
            if depths.shape[0] > 1:
                axis[0, j].scatter(x, y, color='purple', marker='o', label='Point')
            else:
                axis[0].scatter(x, y, color='purple', marker='o', label='Point')

        true_wave = np.roll(Incident[:, 0], int(depths[j]))
        true_wave = (true_wave.transpose() - np.mean(true_wave, axis=0)) / np.std(true_wave, axis=0)
        noise_id_wave = (id.transpose() - np.mean(id, axis=0)) / np.std(id, axis=0)
        noise_ind_wave = (ind.transpose() - np.mean(ind, axis=0)) / np.std(ind, axis=0)
        emitted_wave = (ModFs[:, 0].transpose() - np.mean(ModFs[:, 0], axis=0)) / np.std(ModFs[:, 0], axis=0)

        if depths.shape[0] > 1:
            axis[1, j].plot(true_wave, label='GT Wave')
            axis[1, j].plot(noise_id_wave, color='pink',label='Decoded ID Wave')
            axis[1, j].plot(noise_ind_wave, color='red',  label='Decoded Ind Wave')
            axis[1, j].plot(emitted_wave, label='Emitted Wave')
            axis[1, j].set_title('Phase Shifts')
            axis[1, j].legend()
        else:
            axis[1].plot(true_wave, label='GT Wave')
            axis[1].plot(noise_id_wave, color='pink', label='Decoded ID Wave')
            axis[1].plot(noise_ind_wave, color='red',  label='Decoded Ind Wave')
            axis[1].plot(emitted_wave, label='Emitted Wave')
            axis[1].set_title('Phase Shifts')
            axis[1].legend()


def debug_direct(params,  clean_sim_pulses_pp, clean_sim_pulses_ave,
                 c_vals_pp, pulses_pp, decoded_depths_dtof_maxguass_pp,
                 decoded_depths_dtof_maxguass_ave, c_vals_ave, pulses_ave, depths, t_domain, sigma,
                 peak_power, K, sbr, pAveAmbient, pAveSource):

    meanBeta = params['meanBeta']
    dt = params['dt']
    tau = params['rep_tau']
    n_tbins = params['n_tbins']
    T = params['T']
    tbin_depth_res = direct_tof_utils.time2depth(params['rep_tau'] / params['n_tbins'])

    ## PLOT DIRECT PEAK POWER
    figure, axis = plt.subplots(2, depths.shape[0])
    depths = depths.astype(int)
    for i in range(0, depths.shape[0]):

        src_pp = clean_sim_pulses_pp[depths[i], :]
        src_ave = clean_sim_pulses_ave[depths[i], :]
        photon_count_pp = np.sum(src_pp)
        photon_count_ave = np.sum(src_ave)
        if depths.shape[0] > 1:
            gaus_pp = np.squeeze(c_vals_pp[:, i, :])
        else:
            gaus_pp = np.squeeze(c_vals_pp[i, :])
        plot_pulses_pp = np.transpose(gaus_pp)

        if depths.shape[0] > 1:
            axis[0, i].plot(plot_pulses_pp, label='detected histogram')
        else:
            axis[0].plot(plot_pulses_pp, label='detected histogram')

        clean_pp = pulses_pp.simulate_peak_power(peak_power, pAveSource=pAveSource, num_measures=K,
                                                 n_mc_samples=1, dt=dt, tau=tau, add_noise=False)
        if depths.shape[0] > 1:
            clean_pp = np.transpose(np.squeeze(clean_pp[i, :]))
            axis[0, i].plot(clean_pp, label='true depth')
            est_tshifts_pp = direct_tof_utils.depth2time(decoded_depths_dtof_maxguass_pp[:, i])
        else:
            clean_pp = np.transpose(np.squeeze(clean_pp))
            axis[0].plot(clean_pp, label='true depth')
            est_tshifts_pp = direct_tof_utils.depth2time(decoded_depths_dtof_maxguass_pp[i])

        est_pulses_list_pp = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=est_tshifts_pp, t_domain=t_domain)
        est_pulse_pp = est_pulses_list_pp[0]

        est_pulse_pp.set_mean_beta(meanBeta)
        est_pulse_pp.set_sbr(sbr)
        est_pulse_pp.set_ambient(pAveAmbient)
        est_pulse_pp.set_integration_time(T)
        est_pp = est_pulse_pp.simulate_peak_power(peak_power, pAveSource=pAveSource, num_measures=K, dt=dt, tau=tau,
                                                  add_noise=False)

        if depths.shape[0] > 1:
            axis[0, i].plot(np.transpose(np.squeeze(est_pp)), color='yellow', label='estimated depth')
            axis[0, i].set_title('Direct Peak Power with depth : ' + str(depths[i]*tbin_depth_res) + ' / photon count : ' + str(photon_count_pp))
            axis[0, i].legend()
        else:
            axis[0].plot(np.transpose(np.squeeze(est_pp)), color='yellow', label='estimated depth')
            axis[0].set_title('Direct Peak Power with depth : ' + str(depths[i]*tbin_depth_res) + ' / photon count : ' + str(photon_count_pp))
            axis[0].legend()

        ## PLOT DIRECT AVERAGE POWER
        if (depths.shape[0] > 1):
            gaus_ave = np.squeeze(c_vals_ave[:, i, :])
        else:
            gaus_ave = np.squeeze(c_vals_ave[i, :])
        plot_pulses_ave = np.transpose(gaus_ave)

        if depths.shape[0] > 1:
            axis[1, i].plot(plot_pulses_ave, label='detected histogram')
        else:
            axis[1].plot(plot_pulses_ave, label='detected histogram')

        clean_ave = pulses_ave.simulate_avg_power(pAveSource, n_mc_samples=1, dt=dt, tau=tau, add_noise=False)

        if depths.shape[0] > 1:
            clean_ave = np.transpose(np.squeeze(clean_ave[i, :]))
            axis[1, i].plot(clean_ave, linewidth=2, label='true depth')
            est_tshifts_ave = direct_tof_utils.depth2time(decoded_depths_dtof_maxguass_ave[:, i])

        else:
            clean_ave = np.transpose(np.squeeze(clean_ave))
            axis[1].plot(clean_ave, linewidth=2, label='true depth')
            est_tshifts_ave = direct_tof_utils.depth2time(decoded_depths_dtof_maxguass_ave[i])

        est_pulses_list_ave = tirf.init_gauss_pulse_list(n_tbins, sigma, mu=est_tshifts_ave, t_domain=t_domain)
        est_pulse_ave = est_pulses_list_ave[0]

        est_pulse_ave.set_mean_beta(meanBeta)
        est_pulse_ave.set_sbr(sbr)
        est_pulse_ave.set_ambient(pAveAmbient)
        est_pulse_ave.set_integration_time(T)
        est_ave = est_pulse_ave.simulate_avg_power(pAveSource, n_mc_samples=1, dt=dt, tau=tau, add_noise=False)

        if depths.shape[0] > 1:
            axis[1, i].plot(np.transpose(np.squeeze(est_ave)), color='yellow', label='estimated depth')
            axis[1, i].set_title('Direct Average Power with depth : ' + str(depths[i]*tbin_depth_res) + ' / photon count : ' + str(photon_count_ave))
            axis[1, i].legend()
        else:
            axis[1].plot(np.transpose(np.squeeze(est_ave)), color='yellow', label='estimated depth')
            axis[1].set_title('Direct Average Power with depth : ' + str(depths[i]*tbin_depth_res) + ' / photon count : ' + str(photon_count_ave))
            axis[1].legend()