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

file = np.load('data/results/ntbins_256_monte_1000_lvls_20_exp_0.npz', allow_pickle=True)
file1024 = np.load('data/results/ntbins_1024_monte_1000_lvls_20_exp_0.npz', allow_pickle=True)

#mae_pp = np.load('./data/results/ntbins_1024_monte_10_exp_AddPeak.npz')

mae = file['results'].item()
mae1024 = file1024['results'].item()
sbr_levels = file['sbr_levels']
photons_levels = file['photon_levels']
mae_idtof = mae['mae_itof']
mae_itof = mae['mae_idtof']

mae_itof1024 = mae1024['mae_idtof']
mae_dtof_maxgauss1024 = mae1024['mae_dtof_maxgauss']


mae_dtof_argmax = mae['mae_dtof_argmax']
mae_dtof_maxgauss = mae['mae_dtof_maxgauss']
mae_pulsed_idtof = mae['mae_pulsed_idtof']

diff1 = mae_dtof_maxgauss - mae_itof
diff2 = mae_idtof - mae_itof
#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_idtof, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_itof, cmap=cm.autumn, linewidth=0, antialiased=False)

ax.plot_surface(np.log(sbr_levels), np.log(photons_levels), diff1, cmap=cm.autumn,linewidth=0, antialiased=False)
#ax.plot_surface(np.log(sbr_levels), np.log(photons_levels), diff2, cmap=cm.Blues,linewidth=0, antialiased=False)


#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_dtof_argmax, cmap=cm.summer,linewidth=0, antialiased=False)
#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_dtof_maxgauss, cmap=cm.Purples,linewidth=0, antialiased=False)
#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_dtof_maxgauss1024, cmap=cm.PiYG,linewidth=0, antialiased=False)

#ax.plot_surface(np.log10(sbr_levels), np.log10(photons_levels), mae_pulsed_idtof, cmap=cm.winter,linewidth=0, antialiased=False)


ax.set_xlabel('log10 sbr_levels')
ax.set_ylabel('log10 photon levels')
ax.set_zlabel('mae')

plt.show()
print('helloworld')