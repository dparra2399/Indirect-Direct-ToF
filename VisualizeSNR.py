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
file = np.load('data/results/ntbins_100_monte_5000_pw_1_exp_200.npz', allow_pickle=True)


mae = file['results'].item()
params = file['params'].item()
tbin_res = params['rep_tau'] / params['n_tbins']
tbin_depth_res = direct_tof_utils.time2depth(tbin_res)

ham_d = 0
sin_id = 0
sin_d = 0
ave = 0
gating = 1
hamk3_gated = 1
hamk4_gated = 1
hamk5_gated = 0
hamk3_id = 0
hamk4_id = 0
hamk5_id = 0
identity = 0

sbr_levels = file['sbr_levels']
ambient_levels = file['pAveAmbient_levels']
photons_levels = file['pAveSource_levels']

if identity:
    mae_identity_pp = mae['Identity_PP']
    mae_identity_ave = mae['Identity_AVE']

if gating:
    mae_identity_gating_pp = mae['IntegratedGated_PP']
    mae_identity_gating_ave = mae['IntegratedGated_AVE']

if sin_id:
    mae_sinosoid_idtof = mae['KTapSinusoid_IDTOF']
    mae_sinosoid_itof = mae['KTapSinusoid_ITOF']
    diff_comb_sin = mae_sinosoid_idtof - mae_sinosoid_itof

if sin_d:
    mae_sinosoiud_pp = mae['KTapSinusoid_PP']
    mae_sinosoiud_ave = mae['KTapSinusoid_AVE']

if hamk3_id:
    mae_hamk3_idtof = mae['HamiltonianK3_IDTOF']
    mae_hamk3_itof = mae['HamiltonianK3_ITOF']
    diff_comb_hamk3 = mae_hamk3_idtof - mae_hamk3_itof

if hamk4_id:
    mae_hamk4_idtof = mae['HamiltonianK4_IDTOF']
    mae_hamk4_itof = mae['HamiltonianK4_ITOF']
    diff_comb_hamk4 = mae_hamk4_idtof - mae_hamk4_itof

if hamk5_id:
    mae_hamk5_idtof = mae['HamiltonianK5_IDTOF']
    mae_hamk5_itof = mae['HamiltonianK5_ITOF']
    diff_comb_hamk5 = mae_hamk5_idtof - mae_hamk5_itof

if hamk3_gated:
    mae_hamk3_gated = mae['HamiltonianK3Gated_ITOF']

if hamk3_gated and hamk3_id:
    diff_hamk3 = mae_hamk3_itof - mae_hamk3_gated

if hamk4_gated:
    mae_hamk4_gated = mae['HamiltonianK4Gated_ITOF']

if hamk5_gated:
    mae_hamk5_gated = mae['HamiltonianK5Gated_ITOF']

if ham_d:
    mae_hamk4_pp = mae['HamiltonianK3_PP']
    mae_hamk4_ave = mae['HamiltonianK3_AVE']




fig = plt.figure()
ax = plt.axes(projection='3d')

noise_levels = sbr_levels

if sin_id:
    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_sinosoid_idtof, color='yellow',
                           antialiased=True, label='Sin ID-ToF')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_sinosoid_itof, color='blue',
                            antialiased=True, label='Sin I-ToF')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

if hamk3_id:
    # surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk3_idtof, color='red',
    #                        antialiased=True, label='HamK4 ID-ToF')
    # surf._edgecolors2d = surf._edgecolor3d
    # surf._facecolors2d = surf._facecolor3d

    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk3_itof, color='thistle',
                            antialiased=True, label='HamK3 I-ToF')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

if hamk4_id:
    # surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk4_idtof, color='red',
    #                        antialiased=True, label='HamK4 ID-ToF')
    # surf._edgecolors2d = surf._edgecolor3d
    # surf._facecolors2d = surf._facecolor3d

    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk4_itof, color='thistle',
                            antialiased=True, label='HamK4 I-ToF')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

if hamk5_id:
    # surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk5_idtof, color='red',
    #                        antialiased=True, label='HamK4 ID-ToF')
    # surf._edgecolors2d = surf._edgecolor3d
    # surf._facecolors2d = surf._facecolor3d

    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk5_itof, color='thistle',
                            antialiased=True, label='HamK5 I-ToF')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

if hamk3_gated:
    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk3_gated, color='red',
                           antialiased=False, label='HamK3 SIWSSPAD2')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

if hamk4_gated:
    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk4_gated, color='lightgreen',
                           antialiased=True, label='HamK4 SIWSSPAD2')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

if hamk5_gated:
    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk5_gated, color='blue',
                           antialiased=True, label='HamK5 SIWSSPAD2')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d


if identity:
    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_identity_pp, color='green',
                            antialiased=True, label='FRH Peak Power')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

    if ave:
        surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_identity_ave, color='pink',
                               antialiased=True, label='FRH Avg Power')
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d

if gating:
    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_identity_gating_pp, color='hotpink',
                           antialiased=True, label='FRH GATING Peak Power')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

    if ave:
        surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_identity_gating_ave, color='ivory',
                               antialiased=True, label='FRH GATING Avg Power')
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d



if sin_d:
    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_sinosoiud_pp, color='orange',
                            antialiased=True, label='KTap Sin Peak Power')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

    if ave:
        surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_sinosoiud_ave, color='gray',
                                antialiased=True, label='KTap Sin Avg Power')
        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d

if ham_d:
    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk4_pp, color='darkred',
                            antialiased=True, label='HamK4 Sin Peak Power')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

    if ave:
        surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_hamk4_ave, color='peru',
                                antialiased=True, label='HamK4 Sin Avg Power')

        surf._edgecolors2d = surf._edgecolor3d
        surf._facecolors2d = surf._facecolor3d




ax.set_title('depth errors for all cases with time res: ' + '{:.3g}'.format(tbin_res) + ' and depth res: ' + '{:.3g}'.format(tbin_depth_res * 1000) + ' mm')
ax.set_xlabel('log10 sbr levels')
ax.set_ylabel('log10 pAve photon levels')
ax.set_zlabel('mean absolute error in (mm)')

ax.legend()


if hamk3_id and hamk3_gated:
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_hamk3, color='red',linewidth=0, antialiased=False)

    ax.set_title('difference between gated and convential hamitlonian')
    ax.set_xlabel('log10 sbr levels')
    ax.set_ylabel('log10 pAve photon levels')
    ax.set_zlabel('mae difference in (mm)')
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d
    ax.legend()

plt.show()
#
# if identity and sin_id:
#     diff_direct_pp = mae_identity_pp - mae_sinosoid_idtof
#     fig2 = plt.figure()
#     ax2 = plt.axes(projection='3d')
#
#
#     surfdiff_direct_pp = ax2.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_direct_pp, color='orange',linewidth=0, antialiased=False)
#
#     ax2.set_title('difference between (DIRECT FRH - COMBINED SIN) peak power')
#     ax2.set_xlabel('log10 sbr levels')
#     ax2.set_ylabel('log10 pAve photon levels')
#     ax2.set_zlabel('mae difference in (mm)')
#     surfdiff_direct_pp._edgecolors2d = surfdiff_direct_pp._edgecolor3d
#     surfdiff_direct_pp._facecolors2d = surfdiff_direct_pp._facecolor3d
#     ax2.legend()
#
# if ham_d and hamk3_id:
#     diff_hamk4_pp = mae_hamk4_pp - mae_hamk3_idtof
#     fig3 = plt.figure()
#     ax3 = plt.axes(projection='3d')
#
#     surfdiff_direct_ave = ax3.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_hamk4_pp, color='orange',linewidth=0, antialiased=False)
#
#     ax3.set_title('difference between (DIRECT CODING HAMK4 - COMBINED HAMK4) peak power')
#     ax3.set_xlabel('log10 sbr levels')
#     ax3.set_ylabel('log10 pAve photon levels')
#     ax3.set_zlabel('mae difference in (mm)')
#     surfdiff_direct_ave._edgecolors2d = surfdiff_direct_ave._edgecolor3d
#     surfdiff_direct_ave._facecolors2d = surfdiff_direct_ave._facecolor3d
#     ax3.legend()
#
# if sin_id:
#     fig4 = plt.figure()
#     ax4 = plt.axes(projection='3d')
#
#     surfdiff2 = ax4.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_comb_sin, color='red',linewidth=0, antialiased=False)
#
#     ax4.set_title('difference between (COMBINED SIN - CODING SIN)')
#     ax4.set_xlabel('log10 sbr levels')
#     ax4.set_ylabel('log10 pAve photon levels')
#     ax4.set_zlabel('mae difference in (mm)')
#     surfdiff2._edgecolors2d = surfdiff2._edgecolor3d
#     surfdiff2._facecolors2d = surfdiff2._facecolor3d
#     ax4.legend()
#
# if hamk3_id:
#     fig6 = plt.figure()
#     ax6 = plt.axes(projection='3d')
#
#     surfdiff3 = ax6.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_comb_hamk3, color='red',linewidth=0, antialiased=False)
#
#     ax6.set_title('difference between (COMBINED HAMK4 - INDIRECT HAMK4)')
#     ax6.set_xlabel('log10 sbr levels')
#     ax6.set_ylabel('log10 pAve photon levels')
#     ax6.set_zlabel('mae difference in (mm)')
#     surfdiff3._edgecolors2d = surfdiff3._edgecolor3d
#     surfdiff3._facecolors2d = surfdiff3._facecolor3d
#     ax6.legend()
#
# if sin_id and sin_d:
#     diff_sin_pp = mae_sinosoiud_pp - mae_sinosoid_idtof
#
#     fig5 = plt.figure()
#     ax5 = plt.axes(projection='3d')
#
#     surfdiff3 = ax5.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_sin_pp, color='red',linewidth=0, antialiased=False)
#
#     ax5.set_title('difference between (DIRECT CODING SIN - COMBINED SIN) peak power')
#     ax5.set_xlabel('log10 sbr levels')
#     ax5.set_ylabel('log10 pAve photon levels')
#     ax5.set_zlabel('mae difference in (mm)')
#     surfdiff3._edgecolors2d = surfdiff3._edgecolor3d
#     surfdiff3._facecolors2d = surfdiff3._facecolor3d
#     ax5.legend()
#
# if gating and hamk3_id:
#         diff_gating_hamk4_pp = mae_identity_gating_pp - mae_hamk3_idtof
#
#         fig7 = plt.figure()
#         ax7 = plt.axes(projection='3d')
#
#         surfdiff_gating_ham_pp = ax7.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_gating_hamk4_pp, color='teal',
#                                      linewidth=0, antialiased=False)
#
#         ax7.set_title('difference between (DIRECT FRH GATING - COMBINED SIN) peak power')
#         ax7.set_xlabel('log10 sbr levels')
#         ax7.set_ylabel('log10 pAve photon levels')
#         ax7.set_zlabel('mae difference in (mm)')
#         surfdiff_gating_ham_pp._edgecolors2d = surfdiff_gating_ham_pp._edgecolor3d
#         surfdiff_gating_ham_pp._facecolors2d = surfdiff_gating_ham_pp._facecolor3d
#         ax7.legend()
#
# if gating and hamk3_id and ave:
#         diff_gating_hamk4_ave = mae_identity_gating_ave - mae_hamk3_idtof
#
#         fig8 = plt.figure()
#         ax8 = plt.axes(projection='3d')
#
#         surfdiff_gating_ham_ave = ax8.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_gating_hamk4_ave, color='lightcyan',
#                                      linewidth=0, antialiased=False)
#
#         ax8.set_title('difference between (DIRECT FRH GATING - COMBINED SIN) average power')
#         ax8.set_xlabel('log10 sbr levels')
#         ax8.set_ylabel('log10 pAve photon levels')
#         ax8.set_zlabel('mae difference in (mm)')
#         surfdiff_gating_ham_ave._edgecolors2d = surfdiff_gating_ham_ave._edgecolor3d
#         surfdiff_gating_ham_ave._facecolors2d = surfdiff_gating_ham_ave._facecolor3d
#         ax8.legend()


# Write-up for gating cases
  # SPAD Measurements
  # Plot and figures
  # Argue hyrbid method
  # bullet points
# Prepare slides (summerize argument)
  # Understand and agree
# Study different codes, better than ham?
  # How to do that?
# Implement experiment in hardware
# Ask Alex - several cameras that use this gating (SwissSPAD)
# Ask Felipe - cameras that do ham codes

# PMD - Wikipedia
# Photonic mixer device

# Gating - reduce data usage
# Can only read binary data
# https://www.photonics.com/Articles/PMD_Camera_Enhances_3-D_Imaging/a56859