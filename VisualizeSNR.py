# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from IPython.core import debugger
from direct_toflib import direct_tof_utils
from matplotlib import cm

breakpoint = debugger.set_trace

#this one is peak_power with with 30
file = np.load('data/results/ntbins_1024_monte_2000_pw_30_exp_0.npz', allow_pickle=True)

#mae_pp = np.load('./data/results/ntbins_1024_monte_10_exp_AddPeak.npz')

mae = file['results'].item()
params = file['params'].item()
tbin_res = params['rep_tau'] / params['n_tbins']
tbin_depth_res = direct_tof_utils.time2depth(tbin_res)

sbr_levels = file['sbr_levels']
ambient_levels = file['pAveAmbient_levels']
photons_levels = file['pAveSource_levels']

mae_sinosoid_idtof = mae['KTapSinusoid_IDTOF']
mae_sinosoid_itof = mae['KTapSinusoid_ITOF']
mae_hamk4_idtof = mae['HamiltonianK4_IDTOF']
mae_hamk4_itof = mae['HamiltonianK4_ITOF']

mae_identity_pp = mae['Identity_PP']
mae_identity_ave = mae['Identity_AVE']
mae_sinosoiud_pp = mae['KTapSinusoid_PP']
mae_sinosoiud_ave = mae['KTapSinusoid_AVE']
mae_hamk4_pp = mae['HamiltonianK4_PP']
mae_hamk4_ave = mae['HamiltonianK4_AVE']

diff_direct_pp = mae_identity_pp - mae_sinosoid_idtof
diff_sin_pp = mae_sinosoiud_pp - mae_sinosoid_idtof
diff_hamk4_pp = mae_sinosoiud_pp - mae_hamk4_idtof

diff_comb_hamk4 = mae_hamk4_idtof - mae_hamk4_itof
diff_comb_sin = mae_sinosoid_idtof - mae_sinosoid_itof

fig = plt.figure()
ax = plt.axes(projection='3d')
ham = 1
sin = 0
ave = 0

noise_levels = sbr_levels

if sin:
    surf1 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_sinosoid_idtof, color='yellow',
                           antialiased=True, label='Sin ID-ToF')
    surf1._edgecolors2d = surf1._edgecolor3d
    surf1._facecolors2d = surf1._facecolor3d

    surf2 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_sinosoid_itof, color='blue',
                            antialiased=True, label='Sin I-ToF')
    surf2._edgecolors2d = surf2._edgecolor3d
    surf2._facecolors2d = surf2._facecolor3d

if ham:
    surf3 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk4_idtof, color='red',
                           antialiased=True, label='HamK4 ID-ToF')
    surf3._edgecolors2d = surf3._edgecolor3d
    surf3._facecolors2d = surf3._facecolor3d

    surf4 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk4_itof, color='black',
                            antialiased=True, label='HamK4 I-ToF')
    surf4._edgecolors2d = surf4._edgecolor3d
    surf4._facecolors2d = surf4._facecolor3d


surf5 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_identity_pp, color='green',
                        antialiased=True, label='FRH Peak Power')
surf5._edgecolors2d = surf5._edgecolor3d
surf5._facecolors2d = surf5._facecolor3d

if ave:
    surf6 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_identity_ave, color='pink',
                            antialiased=True, label='FRH Avg Power')
    surf6._edgecolors2d = surf6._edgecolor3d
    surf6._facecolors2d = surf6._facecolor3d

if sin:
    surf7 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_sinosoiud_pp, color='orange',
                            antialiased=True, label='KTap Sin Peak Power')
    surf7._edgecolors2d = surf7._edgecolor3d
    surf7._facecolors2d = surf7._facecolor3d

    if ave:
        surf8 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_sinosoiud_ave, color='gray',
                                antialiased=True, label='KTap Sin Avg Power')
        surf8._edgecolors2d = surf8._edgecolor3d
        surf8._facecolors2d = surf8._facecolor3d

if ham:
    surf9 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk4_pp, color='darkred',
                            antialiased=True, label='HamK4 Sin Peak Power')
    surf9._edgecolors2d = surf9._edgecolor3d
    surf9._facecolors2d = surf9._facecolor3d

    if ave:
        surf10 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk4_ave, color='peru',
                                antialiased=True, label='HamK4 Sin Avg Power')
        surf10._edgecolors2d = surf10._edgecolor3d
        surf10._facecolors2d = surf10._facecolor3d




ax.set_title('depth errors for all cases with time res: ' + '{:.3g}'.format(tbin_res) + ' and depth res: ' + '{:.3g}'.format(tbin_depth_res * 1000) + ' mm')
ax.set_xlabel('log10 sbr levels')
ax.set_ylabel('log10 pAve photon levels')
ax.set_zlabel('mean absolute error in (mm)')

ax.legend()


fig2 = plt.figure()
ax2 = plt.axes(projection='3d')


surfdiff_direct_pp = ax2.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_direct_pp, color='orange',linewidth=0, antialiased=False)

ax2.set_title('difference between (DIRECT FRH - COMBINED SIN) peak power')
ax2.set_xlabel('log10 sbr levels')
ax2.set_ylabel('log10 pAve photon levels')
ax2.set_zlabel('mae difference in (mm)')
surfdiff_direct_pp._edgecolors2d = surfdiff_direct_pp._edgecolor3d
surfdiff_direct_pp._facecolors2d = surfdiff_direct_pp._facecolor3d
ax2.legend()


if ham:
    fig3 = plt.figure()
    ax3 = plt.axes(projection='3d')

    surfdiff_direct_ave = ax3.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_hamk4_pp, color='orange',linewidth=0, antialiased=False)

    ax3.set_title('difference between (DIRECT CODING HAMK4 - COMBINED HAMK4) peak power')
    ax3.set_xlabel('log10 sbr levels')
    ax3.set_ylabel('log10 pAve photon levels')
    ax3.set_zlabel('mae difference in (mm)')
    surfdiff_direct_ave._edgecolors2d = surfdiff_direct_ave._edgecolor3d
    surfdiff_direct_ave._facecolors2d = surfdiff_direct_ave._facecolor3d
    ax3.legend()

if sin:
    fig4 = plt.figure()
    ax4 = plt.axes(projection='3d')

    surfdiff2 = ax4.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_comb_sin, color='red',linewidth=0, antialiased=False)

    ax4.set_title('difference between (COMBINED SIN - CODING SIN)')
    ax4.set_xlabel('log10 sbr levels')
    ax4.set_ylabel('log10 pAve photon levels')
    ax4.set_zlabel('mae difference in (mm)')
    surfdiff2._edgecolors2d = surfdiff2._edgecolor3d
    surfdiff2._facecolors2d = surfdiff2._facecolor3d
    ax4.legend()

if ham:
    fig6 = plt.figure()
    ax6 = plt.axes(projection='3d')

    surfdiff3 = ax6.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_comb_hamk4, color='red',linewidth=0, antialiased=False)

    ax6.set_title('difference between (COMBINED HAMK4 - INDIRECT HAMK4)')
    ax6.set_xlabel('log10 sbr levels')
    ax6.set_ylabel('log10 pAve photon levels')
    ax6.set_zlabel('mae difference in (mm)')
    surfdiff3._edgecolors2d = surfdiff3._edgecolor3d
    surfdiff3._facecolors2d = surfdiff3._facecolor3d
    ax6.legend()

if sin:
    fig5 = plt.figure()
    ax5 = plt.axes(projection='3d')

    surfdiff3 = ax5.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_sin_pp, color='red',linewidth=0, antialiased=False)

    ax5.set_title('difference between (DIRECT CODING SIN - COMBINED SIN) peak power')
    ax5.set_xlabel('log10 sbr levels')
    ax5.set_ylabel('log10 pAve photon levels')
    ax5.set_zlabel('mae difference in (mm)')
    surfdiff3._edgecolors2d = surfdiff3._edgecolor3d
    surfdiff3._facecolors2d = surfdiff3._facecolor3d
    ax5.legend()

plt.show()
print('helloworld')


#Try on mutiple bin sizes ==> Make direct as close to combined as possible
#Complete other combined case
#Try ham. codes

#Next steps showming mathmatically that the ham. codes perform the same with indirect and direct
#Pyimaging --> Gating
#Are there other better coding functions that we can use