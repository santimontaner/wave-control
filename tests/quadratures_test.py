import numpy as np
import pytest
from wavecontrol import quadratures


def integrate_1d(nodes_weights, func):
    return np.sum(nodes_weights[:, 1] * func(nodes_weights[:, 0]))


def integrate_2d(nodes_weights, func):
    return np.sum(nodes_weights[:, 2] * func(nodes_weights[:, 0], nodes_weights[:, 1]))


def linear_func(x):
    return 2 * x + 1


def quadratic_func(x):
    return x**2 - x + 1


def bilinear_func(x, y):
    return x + y


def biquadratic_func(x, y):
    return x**2 + y**2


@pytest.mark.parametrize("func, expected", [
    (linear_func, 2.0),  # Integral over [-1, 1]
    (quadratic_func, 2.0 / 3 + 2)  # Integral over [-1, 1]
])
def test_1d_integration(func, expected):
    calculated = integrate_1d(quadratures.gauss_1d, func)
    assert np.isclose(calculated, expected, atol=1e-6)


@pytest.mark.parametrize("func, expected", [
    (bilinear_func, 2 / 3.0),  # 2 x Integral over lower-left subtriangle of [0,1]^2 (area of [0, 1]^2 is 1 but weights sum 0.5)
    (biquadratic_func, 1 / 3.0)  # 2 x Integral over lower-left subtriangle of [0,1]^2 (area of [0, 1]^2 is 1 but weights sum 0.5)
])
def test_2d_integration(func, expected):
    calculated = integrate_2d(quadratures.gauss_2d, func)
    assert np.isclose(calculated, expected, atol=1e-6)


def test_weights_sum_two_1d():
    assert np.isclose(np.sum(quadratures.gauss_1d[:, 1]), 2.0, atol=1e-6)


def test_weights_sum_one_2d():
    assert np.isclose(np.sum(quadratures.gauss_2d[:, 2]), 1.0, atol=1e-6)
