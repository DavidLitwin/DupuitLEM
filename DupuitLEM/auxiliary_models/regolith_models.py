"""
Author: David Litwin

Date: 19 May 2020
"""

import numpy as np


class RegolithModel:
    """
    Models to update regolith state: topographic__elevation and aquifer_base__elevation
    given uplift and regolith production.
    """

    def __init__(self, grid):
        """
        Initialize a RegolithModel

        Parameters
        -----
        grid: ModelGrid
            Landlab ModelGrid object.
        """
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
        grid: ModelGrid
            Landlab ModelGrid object.
        equilibrium_depth: float
            Constant thickness value.
        uplift_rate: float
            Constant uplift rate.
        """

        super().__init__(grid)
        self.d_eq = equilibrium_depth
        self.U = uplift_rate

    def run_step(self, dt_m):
        """Advance regolith model one step in time, keeping constant thickness.
        uplift topographic__elevation, offset aquifer_base__elevation from
        surface, set water table elevation based on original aquifer thickness.

        Parameters:
        -----
        dt_m: float
            Timestep for update.
        """

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
        grid: ModelGrid
            Landlab ModelGrid object.
        equilibrium_depth: float
            Constant thickness value.
            Default: 1.0
        uplift_rate: float
            Constant uplift rate.
            Default: 1e-12
        std: float
            Standard deviation of the perturbation.
            Default: 0.01
        seed: int
            Seed for perturbation.
            Default: None
        """

        super().__init__(grid)
        self.d_eq = equilibrium_depth
        self.U = uplift_rate
        self.std = std
        self.r = np.random.RandomState(seed)

    def run_step(self, dt_m):
        """Advance regolith model one step in time, keeping constant thickness.
        uplift topographic__elevation and add perturbation, offset
        aquifer_base__elevation from surface, set water table elevation based
        on original aquifer thickness.

        Parameters:
        -----
        dt_m: float
            Timestep for update.
        """

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
        delta = 1e-3,
    ):
        """
        Parameters:
        -----
        grid: ModelGrid
            Landlab ModelGrid object.
        characteristic_depth: float
            Exponential characteristic depth.
            Default: 1.0
        regolith_production_rate: float
            Maximum regolith production rate.
            Default: 2e-12
        uplift_rate: float
            Constant uplift rate.
            Default: 1e-12
        delta: float
            Minimum allowable thickness.
            Default 1e-3
        """

        super().__init__(grid)
        self.U = uplift_rate
        self.d_s = characteristic_depth
        self.w0 = regolith_production_rate
        self.d = delta

    def run_step(self, dt_m):
        """Advance regolith model one step in time. Uplift topographic__elevation
        Calculate new thickness based on production and subtract from the
        elevation surface. Set water table elevation based
        on original aquifer thickness.

        Parameters:
        -----
        dt_m: float
            Timestep for update.
        """

        # first ensure that the thickness is positive to avoid later problems
        self._base[self._elev - self.d <= self._base] = self._elev[self._elev - self.d <= self._base] - self.d 
        h = self._wt - self._base  # aquifer storage, assume const n
        b0 = self._elev - self._base  # current thickness

        # uplift and regolith prod, using analytical sol for new thickness
        self._elev[self._cores] += self.U * dt_m
        self._base[self._cores] = self._elev[self._cores] - self.d_s * np.log(
            (self.d_s * np.exp(b0[self._cores] / self.d_s) + dt_m * self.w0) / self.d_s
        )

        # update water table
        self._wt[:] = self._base + h


class RegolithConstantBaselevel(RegolithModel):
    """
    Aquifer base is not updated (so it stays at same
    position relative to baselevel), and uplift rate is spatially
    uniform and constant in time. The additional permeable material
    that enters above baselevel is presumed saturated, so the water
    table stays at the same position relative to topography.
    """

    def __init__(self, grid, uplift_rate=1e-12):
        """
        Parameters:
        -----
        grid: ModelGrid
            Landlab ModelGrid object.
        uplift_rate: float
            Constant uplift rate.
        """

        super().__init__(grid)
        self.U = uplift_rate

    def run_step(self, dt_m):
        """Advance regolith model one step in time, keeping constant aquifer base.
        uplift topographic__elevation and water table elevation directly.

        Parameters:
        -----
        dt_m: float
            Timestep for update.
        """

        # uplift
        self._elev[self._cores] += self.U * dt_m
        self._wt[self._cores] += self.U * dt_m
