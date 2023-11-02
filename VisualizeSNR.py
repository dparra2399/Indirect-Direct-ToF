# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from IPython.core import debugger
from matplotlib import cm

breakpoint = debugger.set_trace

file = np.load('data/results/ntbins_1024_monte_3000_lvls_30_exp_1.npz', allow_pickle=True)

#mae_pp = np.load('./data/results/ntbins_1024_monte_10_exp_AddPeak.npz')

mae = file['results'].item()
sbr_levels = file['sbr_levels']
photons_levels = file['pAveSource_levels']
mae_idtof = mae['mae_idtof']
mae_itof = mae['mae_itof']
mae_dtof_argmax = mae['mae_dtof_argmax']
mae_dtof_maxgauss = mae['mae_dtof_maxgauss']
mae_pulsed_idtof = mae['mae_pulsed_idtof']

flag = 0
if flag:
    arr = [0, 1, 2, 3]
    # photons_levels = np.delete(photons_levels, obj=arr, axis=1)
    # sbr_levels = np.delete(sbr_levels, obj=arr, axis=1)
    # mae_idtof = np.delete(mae_idtof, obj=arr, axis=1)
    # mae_itof = np.delete(mae_itof, obj=arr, axis=1)
    # mae_dtof_argmax = np.delete(mae_dtof_argmax, obj=arr, axis=1)
    # mae_dtof_maxgauss = np.delete(mae_dtof_maxgauss, obj=arr, axis=1)
    # mae_pulsed_idtof = np.delete(mae_pulsed_idtof, obj=arr, axis=1)

    photons_levels = np.delete(photons_levels, obj=arr, axis=0)
    sbr_levels = np.delete(sbr_levels, obj=arr, axis=0)
    mae_idtof = np.delete(mae_idtof, obj=arr, axis=0)
    mae_itof = np.delete(mae_itof, obj=arr, axis=0)
    mae_dtof_argmax = np.delete(mae_dtof_argmax, obj=arr, axis=0)
    mae_dtof_maxgauss = np.delete(mae_dtof_maxgauss, obj=arr, axis=0)
    mae_pulsed_idtof = np.delete(mae_pulsed_idtof, obj=arr, axis=0)



diff1 = mae_dtof_argmax - mae_itof
diff2 = mae_idtof - mae_itof

# LOOK AT DIFF2 because
#COMPARE the direct vs coding, check felipe papers (Compressive histogram, and fourier)
#MAke sense of why direct doing so much better, and how much better it is doing
#plot the differences between all the differences

#Comapre peak powers and adjust them to see how they compare
#Compare light levels, direct should be somewhat better then this one.

#Compare mized and indirect case and see how much better the mixed is doing, and in what cases.
#Can we write exiciting story if
#mixed is doing better

fig = plt.figure()
ax = plt.axes(projection='3d')

surf = ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_idtof, color='yellow', antialiased=False, label='ID-ToF')
surf2 = ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_itof, color='blue', antialiased=False, label='I-ToF')
surf3 = ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_dtof_argmax, color='green',antialiased=False, label='D-ToF-ARGMAX')
#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_dtof_maxgauss, cmap=cm.Purples,linewidth=0, antialiased=False)

#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_pulsed_idtof, color='pink',linewidth=0, antialiased=True, label='Pulsed-ID-ToF')


ax.set_title('depth errors for all cases')
ax.set_xlabel('log10 sbr_levels')
ax.set_ylabel('log10 pAve photon levels')
ax.set_zlabel('mean absolute error in (mm)')
surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d

surf2._edgecolors2d = surf2._edgecolor3d
surf2._facecolors2d = surf2._facecolor3d

surf3._edgecolors2d = surf3._edgecolor3d
surf3._facecolors2d = surf3._facecolor3d
ax.legend()


fig2 = plt.figure()
ax2 = plt.axes(projection='3d')


surfdiff1 = ax2.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), diff1, color='orange',linewidth=0, antialiased=True)

ax2.set_title('difference between direct - coding')
ax2.set_xlabel('log10 sbr_levels')
ax2.set_ylabel('log10 pAve photon levels')
ax2.set_zlabel('mae difference in (mm)')
surfdiff1._edgecolors2d = surfdiff1._edgecolor3d
surfdiff1._facecolors2d = surfdiff1._facecolor3d
ax.legend()

fig3 = plt.figure()
ax3 = plt.axes(projection='3d')


surfdiff2 = ax3.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), diff2, color='red',linewidth=0, antialiased=True)

ax3.set_title('difference between combined - coding')
ax3.set_xlabel('log10 sbr_levels')
ax3.set_ylabel('log10 pAve photon levels')
ax3.set_zlabel('mae difference in (mm)')
surfdiff2._edgecolors2d = surfdiff2._edgecolor3d
surfdiff2._facecolors2d = surfdiff2._facecolor3d
ax.legend()

plt.show()
print('helloworld')