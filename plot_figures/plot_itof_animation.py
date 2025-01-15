import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from felipe_utils import CodingFunctionsFelipe
from felipe_utils.research_utils.signalproc_ops import circular_corr
matplotlib.use('TkAgg')

font = {'family': 'serif',
            'size': 24}

matplotlib.rc('font', **font)

(modfs, demodfs) = CodingFunctionsFelipe.GetCosCos(1000, 3)
corrfs = circular_corr(modfs, demodfs, axis=0)

fig, axs = plt.subplots(3, 2, figsize=(10, 10))  # 2x2 grid of subplots
x = np.linspace(0, 2 * np.pi, 1000)

lines = []
colors = ['red', 'green', 'orange']
colors_corrfs = ['darkred', 'darkgreen', 'darkorange']
for i in range(3):
    axs[i, 0].plot(x, demodfs[:, i], lw=2, color=colors[i])  # Static demodulation plot
    axs[i, 0].set_xlim(0, 2 * np.pi)
    axs[i, 0].set_ylim(np.min(modfs), np.max(modfs))
    axs[i, 0].set_xticks([0, 2 * np.pi])
    axs[i, 0].set_xticklabels([0, r'$\tau$'])
    axs[i, 0].set_yticks([])
    #axs[i, 0].set_ylabel('Intensity')
    line, = axs[i, 0].plot([], [], lw=2)  # Dynamic line
    lines.append(line)

    line, = axs[i, 1].plot([], [], lw=2)
    axs[i, 1].set_xlim(0, 2 * np.pi)
    axs[i, 1].set_ylim(np.min(corrfs), np.max(corrfs))
    axs[i, 1].set_xticks([0, 2 * np.pi])
    axs[i, 1].set_xticklabels([0, r'd'])
    #axs[i, 1].set_xlabel('Depth')
    #axs[i, 1].set_ylabel('Intensity')
    axs[i, 1].set_yticks([])

    #axs[0, 1].set_title('Correlation Function')
    # Dynamic line for column 2
    lines.append(line)


def init():
    for line in lines:
        line.set_data([], [])
        line.set_color('blue')  # Initial color
    return lines

def update(frame):
    for i in range(3):
        # Update column 1 dynamic plots
        y = np.roll(modfs[:, i], frame * 10)  # Shift over time
        lines[i * 2].set_data(x, y)

        tmp = np.zeros((1000))
        tmp[:frame * 10] = corrfs[:frame * 10, i]
        tmp[tmp == 0] = np.nan
        lines[i * 2 + 1].set_data(x, tmp)
        lines[i * 2 + 1].set_color(colors_corrfs[i])

    return lines


ani = FuncAnimation(
    fig, update, frames=100, init_func=init, blit=True, interval=50
)
ani.save("itof-animation.gif", writer=PillowWriter(fps=20))

plt.show(block=True)