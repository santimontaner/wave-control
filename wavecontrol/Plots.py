import matplotlib.pyplot as plt
import numpy as np

def space_time(y, width, final_time, n_x, n_t):
    x = np.linspace(0, 1, n_x+1)
    t = np.linspace(0, final_time, n_t+1)
    X, TT = np.meshgrid(x, t, indexing='xy')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(X,TT, y)

    ax.set_aspect('equal')
    ax.set_xlim(0,width)
    ax.set_ylim(0,final_time)
    ax.tick_params(axis='both', which='major', pad=10)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(12)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(12)

    ax.set_xlabel('x',fontsize=20)
    ax.set_ylabel('t',fontsize=20)

    plt.show()
