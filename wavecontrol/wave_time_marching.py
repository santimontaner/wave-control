import numpy as np
import math as math


def solve_explicit(initial_pos, initial_vel, boundary_data, f, width, final_time, n_x, n_t):
    if f == None:
        def f(x):
            return np.zeros(x.shape)

    del_x = width / n_x
    del_t = final_time / n_t
    ratio = (del_x / del_t)**2
    y = np.zeros((n_t + 1, n_x + 1))

    y[0, :] = initial_pos
    y[1, :] = initial_pos + del_t * initial_vel
    y[1, 1:-1] += (0.5 / ratio) * (initial_pos[0:-2] - 2 * initial_pos[1:-1] + initial_pos[2:])
    y[:, n_x] = boundary_data

    # Time marching method
    for k in range(2, n_t + 1, 1):
        laplacian = (y[k - 1, 0:n_x - 1] - 2 * y[k - 1, 1:n_x] + y[k - 1, 2:]) / ratio
        y[k, 1:n_x] = 2 * y[k - 1, 1:n_x] - y[k - 2, 1:n_x] + laplacian - f(y[k - 1, 1:n_x]) * del_t**2
    return y


def to_grid(U, K, N):
    u = np.zeros((K + 1, N + 1, 3))
    for k in range(K + 1):
        for n in range(N + 1):
            u[k, n, :] = U[3 * (k * (N + 1) + n):3 * (k * (N + 1) + n) + 3]
    return u
