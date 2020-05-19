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

    def __init__(self,grid,d_eq=1.0,U=1e-12):
        super()__init__(grid)
        self.d_eq = d_eq
        self.U = U

    def run_step(self,dt_m):

        #uplift and regolith production
        self._elev[self._cores] += self.U*dt_m
        self._base[self._cores] = self._elev[self._cores] - self.d_eq

class RegolithExponentialProduction(RegolithRunner):

    def __init__(self,grid,d_s=1,w0=2e-12,U=1e-12):
        super()__init__(grid)
        self.U = U
        self.d_s = d_s
        self.w0 = w0

    def run_step(self,dt_m):

        #uplift and regolith production
        self._elev[self._cores] += self.U*dt_m
        self._base[self._cores] += self.U*dt_m - self.w0*np.exp(-(self._elev[self._cores]-self._base[self._cores])/self.d_s)*dt_m
