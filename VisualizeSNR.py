# Python imports
# Library imports
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from IPython.core import debugger
from matplotlib import cm

breakpoint = debugger.set_trace

fig = plt.figure()
ax = plt.axes(projection='3d')

mae = np.load('./data/results/ntbins_1024_monte_1000_exp_high_snr.npz')

pAveSourceList = mae['pAveSourceList']
pAveAmbientList = mae['pAveAmbientList']
mae_idtof = mae['mae_itof']
mae_itof = mae['mae_idtof']
mae_dtof = mae['mae_dtof']

flag = 0

if flag:
    arr = [0, 1, 2, 3, 4, 5]
    pAveSourceList = np.delete(pAveSourceList, obj=arr, axis=1)
    pAveAmbientList = np.delete(pAveAmbientList, obj=arr, axis=1)
    mae_idtof = np.delete(mae_idtof, obj=arr, axis=1)
    mae_itof = np.delete(mae_itof, obj=arr, axis=1)
    mae_dtof = np.delete(mae_dtof, obj=arr, axis=1)

    pAveSourceList = np.delete(pAveSourceList, obj=arr, axis=0)
    pAveAmbientList = np.delete(pAveAmbientList, obj=arr, axis=0)
    mae_idtof = np.delete(mae_idtof, obj=arr, axis=0)
    mae_itof = np.delete(mae_itof, obj=arr, axis=0)
    mae_dtof = np.delete(mae_dtof, obj=arr, axis=0)

diff = mae_idtof - mae_itof
ax.plot_surface(np.log10(pAveSourceList), np.log10(pAveAmbientList), mae_idtof, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.plot_surface(np.log10(pAveSourceList), np.log10(pAveAmbientList), mae_itof, cmap=cm.autumn, linewidth=0, antialiased=False)
#ax.plot_surface(np.log(pAveSourceList), np.log(pAveAmbientList), diff, cmap=cm.autumn,linewidth=0, antialiased=False)
ax.plot_surface(np.log10(pAveSourceList), np.log10(pAveAmbientList), mae_dtof, cmap=cm.summer,linewidth=0, antialiased=False)

ax.set_xlabel('pAve source')
ax.set_ylabel('pAve ambient')
ax.set_zlabel('mae')

plt.show()
print('helloworld')