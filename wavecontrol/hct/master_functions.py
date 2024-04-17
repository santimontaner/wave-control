import numpy as np
from .. import quadratures as qr


class HctMasterFunctions:

    def __init__(self, deg=[True, True, True, True]):
        if deg[0]:
            self.N = qr.gauss_1d.shape[0]
            self.phi_0_1d = np.zeros((self.N, 1, 3))
            self.phi_1_1d = np.zeros((self.N, 1, 3))
            self.phi_2_1d = np.zeros((self.N, 1, 3))
            self.beta_1d = np.zeros((self.N, 1))
            for j, gauss in enumerate(qr.gauss_1d):
                x = gauss[0]
                y = gauss[1]
                self.phi_0_1d[j] = self._phi_0(x, y)
                self.phi_1_1d[j] = self._phi_1(x, y)
                self.phi_2_1d[j] = self._phi_2(x, y)
                self.beta_1d[j] = self._beta(x, y)
        if deg[1]:
            self.N = qr.gauss_2d.shape[0]
            self.phi_0_2d = np.zeros((self.N, 1, 3))
            self.phi_1_2d = np.zeros((self.N, 1, 3))
            self.phi_2_2d = np.zeros((self.N, 1, 3))
            self.beta_2d = np.zeros((self.N, 1))
            for j, gauss in enumerate(qr.gauss_2d):
                x = gauss[0]
                y = gauss[1]
                self.phi_0_2d[j] = self._phi_0(x, y)
                self.phi_1_2d[j] = self._phi_1(x, y)
                self.phi_2_2d[j] = self._phi_2(x, y)
                self.beta_2d[j] = self._beta(x, y)

        if deg[2]:
            self.N = qr.gauss_1d.shape[0]
            self.d_phi_0 = np.zeros((self.N, 2, 3))
            self.d_phi_1 = np.zeros((self.N, 2, 3))
            self.d_phi_2 = np.zeros((self.N, 2, 3))
            self.d_beta = np.zeros((self.N, 2, 1))
            for j, gauss in enumerate(qr.gauss_1d):
                x = 0.5 * (1 - gauss[0])
                y = 0.5 * (1 + gauss[0])
                self.d_phi_0[j] = self._d_phi_0(x, y)
                self.d_phi_1[j] = self._d_phi_1(x, y)
                self.d_phi_2[j] = self._d_phi_2(x, y)
                self.d_beta[j] = self._d_beta(x, y)
        if deg[3]:
            self.N = qr.gauss_2d.shape[0]
            self.d2_phi_0 = np.zeros((self.N, 3, 3))
            self.d2_phi_1 = np.zeros((self.N, 3, 3))
            self.d2_phi_2 = np.zeros((self.N, 3, 3))
            self.d2_beta = np.zeros((self.N, 3, 1))
            for j, gauss in enumerate(qr.gauss_2d):
                x = gauss[0]
                y = gauss[1]
                self.d2_phi_0[j] = self._d2_phi_0(x, y)
                self.d2_phi_1[j] = self._d2_phi_1(x, y)
                self.d2_phi_2[j] = self._d2_phi_2(x, y)
                self.d2_beta[j] = self._d2_beta(x, y)

    def _phi_0(self, x, y):
        return ((1 - x - y)**2) * np.array([[1 + 2 * x + 2 * y, x, y]])

    def _phi_1(self, x, y):
        return np.array([[x**2 * (3 - 2 * x), x**2 * (x - 1), y * x**2]])

    def _phi_2(self, x, y):
        return np.array([[y**2 * (3 - 2 * y), y**2 * x, (y - 1) * y**2]])

    def _beta(self, x, y):
        return x * y * (1 - x - y)

    # First derivatives of master functions
    def _d_phi_0(self, x, y):
        return np.array([[6 * (x + y) * (x + y - 1), (x + y - 1) * (3 * x + y - 1), 2 * y * (x + y - 1)],
                         [6 * (x + y) * (x + y - 1), 2 * x * (x + y - 1), (x + y - 1) * (x + 3 * y - 1)]])

    def _d_phi_1(self, x, y):
        return np.array([[6 * x * (-x + 1), x * (3 * x - 2), 2 * x * y],
                        [0, 0, x**2]])

    def _d_phi_2(self, x, y):
        return np.array([[0, y**2, 0],
                        [6 * y * (-y + 1), 2 * x * y, y * (3 * y - 2)]])

    def _d_beta(self, x, y):
        return np.array([y * (-2 * x - y + 1), x * (-x - 2 * y + 1)]).reshape((2, 1))

    # Second derivatives of master functions
    def _d2_phi_0(self, x, y):
        return np.array([[-6 + 12 * x + 12 * y, -4 + 6 * x + 4 * y, 2 * y],
                         [-6 + 12 * x + 12 * y, 2 * x, -4 + 4 * x + 6 * y],
                         [-6 + 12 * x + 12 * y, -2 + 4 * x + 2 * y, -2 + 2 * x + 4 * y]])

    def _d2_phi_1(self, x, y):
        return np.array([[6 - 12 * x, -2 + 6 * x, 2 * y],
                         [0, 0, 0],
                         [0, 0, 2 * x]])

    def _d2_phi_2(self, x, y):
        return np.array([[0, 0, 0],
                         [6 - 12 * y, 2 * x, -2 + 6 * y],
                         [0, 2 * y, 0]])

    def _d2_beta(self, x, y):
        return np.array([-2 * y, -2 * x, 1 - 2 * (x + y)]).reshape((3, 1))
