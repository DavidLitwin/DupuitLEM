"""
description
"""

import numpy as np

class RegolithRunner:

    def __init__(self,grid):

        self._elev = grid.at_node['topographic__elevation']
        self._base = grid.at_node['aquifer_base__elevation']
        self._cores = grid.core_nodes

    def run_step(self):
        raise NotImplementedError

class RegolithConstantThickness(RegolithRunner):

    def __init__(self,grid,equilibrium_depth=1.0,uplift_rate=1e-12):
        super().__init__(grid)
        self.d_eq = equilibrium_depth
        self.U = uplift_rate

    def run_step(self,dt_m):

        #uplift and regolith production
        self._elev[self._cores] += self.U*dt_m
        self._base[self._cores] = self._elev[self._cores] - self.d_eq

class RegolithExponentialProduction(RegolithRunner):

    def __init__(self,grid,characteristic_depth=1,regolith_production_rate=2e-12,uplift_rate=1e-12):
        super().__init__(grid)
        self.U = uplift_rate
        self.d_s = characteristic_depth
        self.w0 = regolith_production_rate

    def run_step(self,dt_m):

        #uplift and regolith production
        self._elev[self._cores] += self.U*dt_m
        self._base[self._cores] += self.U*dt_m - self.w0*np.exp(-(self._elev[self._cores]-self._base[self._cores])/self.d_s)*dt_m
