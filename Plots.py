import matplotlib.pyplot as plt
import numpy as np

######################################
######    Space-time plot      #######
######################################

def SpaceTimePlot(y, L, T, N, K):
    x = np.linspace(0,1,N+1)
    t = np.linspace(0,T,K+1)
    X, TT = np.meshgrid(x,t,indexing='xy')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolor(X,TT, y)

    ax.set_aspect('equal')
    ax.set_xlim(0,L)
    ax.set_ylim(0,T)
    ax.tick_params(axis='both', which='major', pad=10)

    for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(12)

    for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(12)

    ax.set_xlabel('x',fontsize=20)
    ax.set_ylabel('t',fontsize=20)

    plt.show()
