# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from IPython.core import debugger
from matplotlib import cm

breakpoint = debugger.set_trace

#this one is peak_power with with 30
file = np.load('data/results/ntbins_1024_monte_4000_pw_30_exp_2.npz', allow_pickle=True)
#file = np.load('data/results/ntbins_2000_monte_5000_pw_1_exp_0.npz', allow_pickle=True)
#file = np.load('data/results/ntbins_2000_monte_5000_pw_1_exp_1.npz', allow_pickle=True)

#mae_pp = np.load('./data/results/ntbins_1024_monte_10_exp_AddPeak.npz')

mae = file['results'].item()
sbr_levels = file['sbr_levels']
ambient_levels = file['pAveAmbient_levels']
photons_levels = file['pAveSource_levels']
mae_idtof = mae['mae_idtof']
mae_itof = mae['mae_itof']
mae_dtof_argmax_pp = mae['mae_dtof_argmax_pp']
mae_dtof_maxgauss_pp = mae['mae_dtof_maxgauss_pp']
mae_pulsed_idtof_pp = mae['mae_pulsed_idtof_pp']
mae_dtof_argmax_ave = mae['mae_dtof_argmax_ave']
mae_dtof_maxgauss_ave = mae['mae_dtof_maxgauss_ave']
mae_pulsed_idtof_ave = mae['mae_pulsed_idtof_ave']

diff_direct_pp = mae_dtof_maxgauss_pp - mae_itof
diff_direct_ave = mae_dtof_maxgauss_ave - mae_itof
diff_comb = mae_idtof - mae_itof

#Direct can fail badly
#For the pulse we have different peaks which can be read
#At that value at 60 mm I getting useful data.
#Looks at the singals and see what happens
#Understand why something is wrong and why it crosses
#plotting the error to see at 60
#The pulse width must 30 because that is the best we can do!

#FIRST !!!!!
#Optimize bin size. IF we make them small enough #Done
#Make bin size the width of the pulse. # Done

#Compare gaussian becuase we fitting #Done
#Pulse wdith between 1 and 5 nanoseconds #Done

flag = 0
if flag:
    arr = [0, 1, 2, 3, 4, 5]
    # photons_levels = np.delete(photons_levels, obj=arr, axis=1)
    # sbr_levels = np.delete(sbr_levels, obj=arr, axis=1)
    # mae_idtof = np.delete(mae_idtof, obj=arr, axis=1)
    # mae_itof = np.delete(mae_itof, obj=arr, axis=1)
    # mae_dtof_argmax = np.delete(mae_dtof_argmax, obj=arr, axis=1)
    # mae_dtof_maxgauss = np.delete(mae_dtof_maxgauss, obj=arr, axis=1)
    # mae_pulsed_idtof = np.delete(mae_pulsed_idtof, obj=arr, axis=1)

    photons_levels = np.delete(photons_levels, obj=arr, axis=0)
    sbr_levels = np.delete(sbr_levels, obj=arr, axis=0)
    #ambient_levels = np.delete(ambient_levels, obj=arr, axis=0)
    mae_idtof = np.delete(mae_idtof, obj=arr, axis=0)
    mae_itof = np.delete(mae_itof, obj=arr, axis=0)
    mae_dtof_argmax_pp = np.delete(mae_dtof_argmax_pp, obj=arr, axis=0)
    mae_dtof_maxgauss_pp = np.delete(mae_dtof_maxgauss_pp, obj=arr, axis=0)
    mae_pulsed_idtof_pp = np.delete(mae_pulsed_idtof_pp, obj=arr, axis=0)
    mae_dtof_argmax_ave = np.delete(mae_dtof_argmax_ave, obj=arr, axis=0)
    mae_dtof_maxgauss_ave = np.delete(mae_dtof_maxgauss_ave, obj=arr, axis=0)
    mae_pulsed_idtof_ave = np.delete(mae_pulsed_idtof_ave, obj=arr, axis=0)
    diff_direct_ave = np.delete(diff_direct_ave, obj=arr, axis=0)
    diff_comb = np.delete(diff_comb, obj=arr, axis=0)
    diff_direct_pp = np.delete(diff_direct_pp, obj=arr, axis=0)


fig = plt.figure()
ax = plt.axes(projection='3d')


noise_levels = sbr_levels

surf = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_idtof, color='yellow', antialiased=True, label='ID-ToF')
surf2 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_itof, color='blue', antialiased=True, label='I-ToF')
surf3 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_dtof_maxgauss_pp, color='green',antialiased=True, label='D-ToF-ARGMAX Peak Power')
surf4 = ax.plot_surface(np.log10(noise_levels), np.log10(photons_levels), mae_dtof_maxgauss_ave, color='pink', antialiased=True, label='D-ToF-ARGMAX Avg Power')
#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_dtof_maxgauss, cmap=cm.Purples,linewidth=0, antialiased=False)

#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_pulsed_idtof, color='pink',linewidth=0, antialiased=True, label='Pulsed-ID-ToF')


ax.set_title('depth errors for all cases')
ax.set_xlabel('log10 sbr levels')
ax.set_ylabel('log10 pAve photon levels')
ax.set_zlabel('mean absolute error in (mm)')

surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d

surf2._edgecolors2d = surf2._edgecolor3d
surf2._facecolors2d = surf2._facecolor3d

surf3._edgecolors2d = surf3._edgecolor3d
surf3._facecolors2d = surf3._facecolor3d

surf4._edgecolors2d = surf4._edgecolor3d
surf4._facecolors2d = surf4._facecolor3d
ax.legend()


fig2 = plt.figure()
ax2 = plt.axes(projection='3d')


surfdiff_direct_pp = ax2.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_direct_pp, color='orange',linewidth=0, antialiased=True)

ax2.set_title('difference between (DIRECT - CODING) peak power')
ax2.set_xlabel('log10 sbr levels')
ax2.set_ylabel('log10 pAve photon levels')
ax2.set_zlabel('mae difference in (mm)')
surfdiff_direct_pp._edgecolors2d = surfdiff_direct_pp._edgecolor3d
surfdiff_direct_pp._facecolors2d = surfdiff_direct_pp._facecolor3d
ax2.legend()


fig3 = plt.figure()
ax3 = plt.axes(projection='3d')

surfdiff_direct_ave = ax3.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_direct_ave, color='orange',linewidth=0, antialiased=True)

ax3.set_title('difference between (DIRECT - CODING) avg power')
ax3.set_xlabel('log10 sbr levels')
ax3.set_ylabel('log10 pAve photon levels')
ax3.set_zlabel('mae difference in (mm)')
surfdiff_direct_ave._edgecolors2d = surfdiff_direct_ave._edgecolor3d
surfdiff_direct_ave._facecolors2d = surfdiff_direct_ave._facecolor3d
ax3.legend()

fig4 = plt.figure()
ax4 = plt.axes(projection='3d')

surfdiff2 = ax4.plot_surface(np.log10(noise_levels), np.log10(photons_levels), diff_comb, color='red',linewidth=0, antialiased=True)

ax4.set_title('difference between combined - coding')
ax4.set_xlabel('log10 ambient levels')
ax4.set_ylabel('log10 pAve photon levels')
ax4.set_zlabel('mae difference in (mm)')
surfdiff2._edgecolors2d = surfdiff2._edgecolor3d
surfdiff2._facecolors2d = surfdiff2._facecolor3d
ax4.legend()

plt.show()
print('helloworld')