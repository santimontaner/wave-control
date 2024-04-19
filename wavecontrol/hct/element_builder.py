# Santiago Montaner, 2018

# References
# [1]
# Author: Arnd Meyer
# Title: A simplified calculation of reduced HCT-basis in FE context
# Journal: Computational Methods in Applied Mathematics
# [2]
# Author: David Dunavant,
# Title: High Degree Efficient Symmetrical Gaussian Quadrature Rules for the Triangle,
# Journal International Journal for Numerical Methods in Engineering,

import numpy as np
from numpy.linalg import norm
import numpy.typing as npt
from .. import mesh as mesh
from .. import quadratures as qr
from .master_functions import HctMasterFunctions


class HctElementMatrixBuilder:

    def __init__(self, vertices: npt.ArrayLike, evaluation: HctMasterFunctions):
        """
        Initialize the HctElementMatrixBuilder object with the given vertices and evaluation object.

        This constructor initializes various properties of the HCT element matrix builder, including
        the mesh barycenter, exterior edges, Jacobian matrix, rotation matrix, outward normals, and
        pre-allocates arrays for further computations.

        Parameters:
        - vertices (npt.ArrayLike): A 3x2 array-like structure containing the coordinates of the three vertices
        defining an element. These vertices are used to compute the geometry and properties of the element.
        - evaluation: An object that provides methods for evaluating certain functions or operations required
        in the construction of the element matrices. The specific requirements for this object are determined
        by the implementation details of the HctElementMatrixBuilder.

        The constructor also calls internal methods to initialize interior edges and other necessary properties
        for building the element matrices.
        """
        self.ev = evaluation
        self._vertices = vertices
        self._mesh_barycenter = np.mean(vertices, 0)

        self._tri_edges = self._calculate_tri_edges(self._vertices)
        self._in_normals = self._calculate_interior_normals()
        self._out_normals = self._tri_edges @ np.array([[0, -1], [1, 0]]).T

        self._jacobian_mat = np.column_stack(
            (self._tri_edges[2], -self._tri_edges[1]))
        self._mu = np.abs(np.linalg.det(self._jacobian_mat)) / 3

        self._initialize()

    def build_interior(self) -> np.ndarray:
        number_of_nodes = qr.gauss_2d.shape[0]
        quad_weights = np.zeros((number_of_nodes, 1))
        quad_weights[:, 0] = qr.gauss_2d[:, 2]

        K = np.zeros((9, 9))

        for k in range(3):
            kp, km = self._rotate_index(k)
            J, H = self._compute_J_H(kp, km)

            # a= tau1, b= tau2, c= tau3, d= tau4 in notation of reference [1]
            a, b, c, d = J[0, 0], J[0, 1], J[1, 0], J[1, 1]

            # G matrix for the wave control problem
            G = np.array([b**2 - d**2, a**2 - c**2, 2 * (c * d - a * b)]) / (self._mu**2)
            DD = self._calculate_hessian(k, km, kp, H)

            wave_operator = G @ DD
            wave_op_integral = np.multiply(quad_weights, wave_operator)
            K += self._mu * 0.5 * np.tensordot(wave_op_integral, wave_operator, axes=(0, 0))
        return K

    def build_boundary(self, edge_tri_idx):
        """
        # `edge_tri_idx` is the subtriangle where the edge is located
        """
        N = qr.gauss_1d.shape[0]
        quad_weights = np.zeros((N, 1))
        quad_weights[:, 0] = qr.gauss_1d[:, 1]

        C = np.array([1, 0])
        kp, km = self._rotate_index(edge_tri_idx)
        J, H = self._compute_J_H(kp, km)

        D = np.zeros((N, 2, 9))
        D[:, :, 3 * edge_tri_idx:3 * edge_tri_idx + 3] = self.ev.d_phi_0 @ H @ self._M[edge_tri_idx]
        D[:, :, 3 * kp:3 * kp + 3] = self.ev.d_phi_1 @ H + self.ev.d_phi_0 @ H @ self._M[kp] + self.ev.d_beta @ self._b[edge_tri_idx, kp].T
        D[:, :, 3 * km:3 * km + 3] = self.ev.d_phi_2 @ H + self.ev.d_phi_0 @ H @ self._M[km] + self.ev.d_beta @ self._b[edge_tri_idx, km].T

        G = np.linalg.inv(J).T
        D = C @ G @ D

        w = np.sqrt(np.dot(self._tri_edges[edge_tri_idx, :], self._tri_edges[edge_tri_idx, :])) * 0.5
        wD = np.multiply(quad_weights, D)
        return w * np.tensordot(wD, D, axes=(0, 0))

    def build_init_pos(self, k=2):
        # on the bottom boundary we always integrate over the 3rd subtriangle,
        # which corresponds to k=2.
        L = np.zeros((9, 3))
        kp, km = self._rotate_index(k)

        J, H = self._compute_J_H(kp, km)
        quad_weights = (qr.gauss_1d + 1) * 0.5

        G = np.linalg.inv(J).T
        C = np.array([[0, 1]])

        for j, gauss in enumerate(quad_weights):
            x, y = gauss[0], 1 - gauss[0]
            D = np.zeros((2, 9))
            D0 = self._calc_D_matrices(k, kp, km, H, j, x, y, D, self.ev.d_phi_0, self.ev.d_phi_1, self.ev.d_phi_2, self.ev.d_beta)
            w = 0.5 * norm(self._tri_edges[k, :]) * gauss[1]
            L += w * ((C @ G @ D).T @ D0)
        return L

    def build_init_vel(self, k=2):
        # on the bottom boundary we always integrate over the 3rd subtriangle,
        # which corresponds to k=2.
        L = np.zeros((9, 3))
        kp, km = self._rotate_index(k)

        _, H = self._compute_J_H(kp, km)
        quad_weights = (qr.gauss_1d + 1) * 0.5

        for j, gauss in enumerate(quad_weights):
            x, y = gauss[0], 1 - gauss[0]
            D0 = np.zeros((1, 9))
            DP1 = self._calc_D_matrices(k, kp, km, H, j, x, y, D0, self.ev.phi_0_1d, self.ev.phi_1_1d, self.ev.phi_2_1d, self.ev.beta_1d)
            w = 0.5 * norm(self._tri_edges[k, :]) * gauss[1]
            L += w * (D0.T @ DP1)
        return L

    def _calc_D_matrices(self, k, kp, km, H, j, x, y, D0, phi_0, phi_1, phi_2, beta):
        D0[:, 3 * k:3 * k + 3] = phi_0[j] @ H @ self._M[k]
        D0[:, 3 * kp:3 * kp + 3] = phi_1[j] @ H + phi_0[j] @ H @ self._M[kp] + beta[j] @ self._b[k, kp].T
        D0[:, 3 * km:3 * km + 3] = phi_2[j] @ H + phi_0[j] @ H @ self._M[km] + beta[j] @ self._b[k, km].T
        D1 = np.zeros((1, 3))
        D1[0, :] = np.array([(1 - x - y), (1 + 2 * x - y), (1 - x + 2 * y)]) / 3.
        return D1

    @ staticmethod
    def _calculate_tri_edges(vertices):
        tri_edges = np.zeros((3, 2))
        tri_edges[0, :] = vertices[2, :] - vertices[1, :]
        tri_edges[1, :] = vertices[0, :] - vertices[2, :]
        tri_edges[2, :] = vertices[1, :] - vertices[0, :]
        return tri_edges

    def _calculate_interior_normals(self):
        in_normals = np.zeros((3, 2))
        for j in range(3):
            jp, jm = self._rotate_index(j)
            in_normals[j, :] = (self._tri_edges[jp, :] - self._tri_edges[jm, :]) / 3
        return in_normals

    @ staticmethod
    def _rotate_index(i):
        return (i + 1) % 3, (i - 1) % 3

    def _initialize(self):
        self._b = np.zeros((3, 3, 3, 1))
        self._M = np.zeros((3, 3, 3))
        t = np.zeros((3, 3))
        T = np.zeros((3, 3, 3))
        S = np.zeros((3, 3))
        for k in range(3):
            S[k, 0] = 3
            S[k, 1:] = self._in_normals[k, :]
        S = 6 * S

        for k in range(3):
            kp, km = self._rotate_index(k)
            sq_of_norm = np.dot(self._tri_edges[k, :], self._tri_edges[k, :])
            self._b[k, kp, 0, 0] = 6 * np.dot(self._tri_edges[k, :], self._in_normals[km, :]) / sq_of_norm
            self._b[k, kp, 1:, 0] = 2 * self._in_normals[km, :] + (3 * self._mu / sq_of_norm) * self._out_normals[k, :]

            self._b[k, km, 0, 0] = -6 * np.dot(self._tri_edges[k, :], self._in_normals[kp, :]) / sq_of_norm
            self._b[k, km, 1:, 0] = 2 * self._in_normals[kp, :] + (3 * self._mu / sq_of_norm) * self._out_normals[k, :]

        for j in range(3):
            jp, jm = self._rotate_index(j)
            t[jm, :] = self._b[jp, j, :, 0]
            t[jp, :] = self._b[jm, j, :, 0]
            t[j, :] = t[jm, :] + t[jp, :]
            t[j, 0] += 6
            t[j, 1:] += -2 * self._in_normals[j, :]
            for k in range(3):
                T[j, k, :] = t[k, :]
            self._M[j] = np.linalg.inv(S) @ T[j]

    def _calculate_hessian(self, k, km, kp, H):
        number_of_nodes = qr.gauss_2d.shape[0]
        DD = np.zeros((number_of_nodes, 3, 9))
        DD1 = self.ev.d2_phi_0 @ H @ self._M[k]
        DD2 = self.ev.d2_phi_1 @ H + self.ev.d2_phi_0 @ H @ self._M[kp] + self.ev.d2_beta @ self._b[k, kp].T
        DD3 = self.ev.d2_phi_2 @ H + self.ev.d2_phi_0 @ H @ self._M[km] + self.ev.d2_beta @ self._b[k, km].T
        DD[:, :, 3 * k:3 * k + 3] = DD1
        DD[:, :, 3 * kp:3 * kp + 3] = DD2
        DD[:, :, 3 * km:3 * km + 3] = DD3
        return DD

    def _compute_J_H(self, kp, km):
        J = np.column_stack((self._in_normals[kp, :], self._in_normals[km, :]))
        H = np.zeros((3, 3))
        H[0, 0] = 1
        H[1:, 1:] = J.T
        return J, H
