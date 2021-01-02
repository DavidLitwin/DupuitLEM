"""
Models to update regolith state: topographic__elevation and aquifer_base__elevation
given uplift and regolith production.

Author: David Litwin

Date: 19 May 2020
"""

import numpy as np


class RegolithModel:
    def __init__(self, grid):

        self._elev = grid.at_node["topographic__elevation"]
        self._base = grid.at_node["aquifer_base__elevation"]
        self._wt = grid.at_node["water_table__elevation"]
        self._cores = grid.core_nodes

    def run_step(self):
        raise NotImplementedError


class RegolithConstantThickness(RegolithModel):
    """
    Constant thickness of regoltih is maintained throughout, and uplift
    rate is spatially uniform and constant in time.
    """

    def __init__(self, grid, equilibrium_depth=1.0, uplift_rate=1e-12):
        """
        Parameters:
        -----
        grid: landlab grid
        equilibrium_depth: float. Constant thickness value
        uplift_rate: float. Constant uplift rate
        """

        super().__init__(grid)
        self.d_eq = equilibrium_depth
        self.U = uplift_rate

    def run_step(self, dt_m):

        # aquifer storage, assume porosity constant
        h = self._wt - self._base

        # uplift and regolith production
        self._elev[self._cores] += self.U * dt_m
        self._base[self._cores] = self._elev[self._cores] - self.d_eq

        # update water table
        self._wt[:] = self._base + h


class RegolithConstantThicknessPerturbed(RegolithModel):
    """
    Constant thickness of regoltih is maintained throughout, and uplift
    rate is spatially uniform and constant in time. A small normally
    distributed perturbation is added to topography each step so channels
    don't simply form parallel rills.
    """

    def __init__(
        self, grid, equilibrium_depth=1.0, uplift_rate=1e-12, std=0.01, seed=None
    ):
        """
        Parameters:
        -----
        grid: landlab grid
        equilibrium_depth: float. Constant thickness value
        uplift_rate: float. Constant uplift rate
        std: float. standard deviation of the perturbation
        seed: int. seed for perturbation
        """

        super().__init__(grid)
        self.d_eq = equilibrium_depth
        self.U = uplift_rate
        self.std = std
        self.r = np.random.RandomState(seed)

    def run_step(self, dt_m):

        # aquifer storage, assume porosity constant
        h = self._wt - self._base

        # uplift and regolith production
        self._elev[self._cores] += self.U * dt_m + self.std * self.r.randn(
            len(self._cores)
        )
        self._base[self._cores] = self._elev[self._cores] - self.d_eq

        # update water table
        self._wt[:] = self._base + h


class RegolithExponentialProduction(RegolithModel):
    """
    Exponential regolith production model, where production rate is a function
    of current thickness. Uplift is spatially uniform and constant in time.
    """

    def __init__(
        self,
        grid,
        characteristic_depth=1,
        regolith_production_rate=2e-12,
        uplift_rate=1e-12,
    ):
        """
        Parameters:
        -----
        grid: landlab grid
        regolith_production_rate: float. maximum regolith production rate
        characteristic_depth: float. exponential characteristic depth
        uplift_rate: float. Constant uplift rate
        """

        super().__init__(grid)
        self.U = uplift_rate
        self.d_s = characteristic_depth
        self.w0 = regolith_production_rate

    def run_step(self, dt_m):

        h = self._wt - self._base  # aquifer storage, assume const n
        b0 = self._elev - self._base  # current thickness

        # uplift and regolith prod, using analytical sol for new thickness
        self._elev[self._cores] += self.U * dt_m
        self._base[self._cores] = self._elev[self._cores] - self.d_s * np.log(
            (self.d_s * np.exp(b0[self._cores] / self.d_s) + dt_m * self.w0) / self.d_s
        )

        # update water table
        self._wt[:] = self._base + h
