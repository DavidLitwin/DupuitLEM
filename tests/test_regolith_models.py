"""
Tests of DupuitLEM regolith models.

Date: 8 Oct 2020
"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from landlab import HexModelGrid, RasterModelGrid
from DupuitLEM.auxiliary_models import (
    RegolithConstantThickness,
    RegolithConstantThicknessPerturbed,
    RegolithExponentialProduction,
)


def test_const_thickness():
    """
    Test RegolithConstantThickness and ensure fields are updated
    based on an exact solution.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)

    z = mg.add_ones("node", "topographic__elevation")
    zb = mg.add_zeros("node", "aquifer_base__elevation")
    zwt = mg.add_ones("node", "water_table__elevation")

    rm = RegolithConstantThickness(mg)
    rm.run_step(1e9)

    assert_almost_equal(z[4], 1.001)
    assert_almost_equal(zb[4], 0.001)
    assert_almost_equal(zwt[4], 1.001)


def test_const_thickness_perturbed():
    """
    Test RegolithConstantThicknessPerturbed and ensure fields are updated
    based on an exact solution.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)

    z = mg.add_ones("node", "topographic__elevation")
    zb = mg.add_zeros("node", "aquifer_base__elevation")
    zwt = mg.add_ones("node", "water_table__elevation")

    rm = RegolithConstantThicknessPerturbed(mg, seed=1)
    rm.run_step(1e9)

    np.random.seed(1)
    x = np.random.randn(1)

    assert_almost_equal(z[4], 1.0 + 0.001 + 0.01 * x)
    assert_almost_equal(zb[4], 0.001 + 0.01 * x)
    assert_almost_equal(zwt[4], 1.0 + 0.001 + 0.01 * x)


def test_exp_reg_prod():
    """
    Test RegolithExponentialProduction and ensure fields are updated
    based on an exact solution.
    """

    mg = RasterModelGrid((3, 3), xy_spacing=10.0)

    z = mg.add_ones("node", "topographic__elevation")
    zb = mg.add_zeros("node", "aquifer_base__elevation")
    zwt = mg.add_ones("node", "water_table__elevation")

    rm = RegolithExponentialProduction(mg)
    rm.run_step(1e9)

    assert_almost_equal(z[4], 1.001)
    assert_almost_equal(zb[4], 1.001 - np.log(np.exp(1) + 1e9 * 2e-12))
    assert_almost_equal(zwt[4], 1.001 - np.log(np.exp(1) + 1e9 * 2e-12) + 1.0)

    b0 = (z - zb)[4]
    rm.run_step(1e9)

    assert_almost_equal(z[4], 1.002)
    assert_almost_equal(zb[4], 1.002 - np.log(np.exp(b0) + 1e9 * 2e-12))
    assert_almost_equal(zwt[4], 1.002 - np.log(np.exp(b0) + 1e9 * 2e-12) + 1.0)
