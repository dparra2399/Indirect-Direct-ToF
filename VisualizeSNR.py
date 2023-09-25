# Python imports
import math
# Library imports
import numpy as np
from scipy import signal
from scipy import special
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from IPython.core import debugger
from matplotlib import cm
import time

import simulate_tof_scene

breakpoint = debugger.set_trace

import CodingFunctions
import Utils
import FileUtils


fig = plt.figure()
ax = plt.axes(projection='3d')

mae = np.load('./data/ntbins_1024_k_4_coding_coscos_monte_1000_exp_20.npz')
direct = np.load('./data/direct_ntbins_1024_monte_1000_exp_0.npz')

pAveSourceList = mae['pAveSourceList']
pAveAmbientList = mae['pAveAmbientList']
mae_idtof = mae['SNR_IDTOF']
mae_itof = mae['SNR_ITOF']

#pAveSourceList = direct['pAveSourceList']
#pAveAmbientList = direct['pAveAmbientList']
mae_dtof = direct['mae']

arr = []
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