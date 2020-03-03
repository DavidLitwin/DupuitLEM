# -*- coding: utf-8 -*-
"""
Created on 25 Feb 2020

This model uses the effective area approch described by Barnhart et al 2019 for
steady variable source area hydrology (see terrainbento BasicVs) along with linear
diffusion and detachment limited erosion. The model assumes an exponential
distribution of recharge rates, where the parameter used is the mean value.
This script tests the model with varying hydraulic conductivity.

@author: dgbli
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from landlab import RasterModelGrid, imshow_grid
from landlab.components import (
    GroundwaterDupuitPercolator,
    FlowAccumulator,
    FastscapeEroder,
    LinearDiffuser,
    LakeMapperBarnes,
    DepressionFinderAndRouter,
)


# Set parameters
uplift_rate = 1E-4/(365*24*3600) # uniform uplift [m/s]
m = 0.5 #Exponent on Q []
n = 1.0 #Exponent on S []
K = 5E-8 #erosivity coefficient [m-1/2 s−1/2]
D = 0.01/(365*24*3600) # hillslope diffusivity [m2/s]
R = 1.5/(365*24*3600)  # annual recharge rate [m/s]
Ks = 1/(3600)  # hydraulic conductivity at the surface [m/s]
w0 = 2E-4/(365*24*3600) #max rate of soil production [m/s]
d_i_rel = 1.0 # initial depth relative to steady state depth [-]
d_s = 1.5 # characteristic soil production depth [m]
d_i = d_i_rel*(-d_s*np.log(uplift_rate/w0)) # initial permeable thickness
T = 5e5*(365*24*3600) # total simulation time [s]
dt = 10*(365*24*3600)
N = T//dt
N = int(N)

#initialize grid and fields
np.random.seed(2)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
cores = grid.core_nodes
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = d_i + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
recharge = grid.add_zeros('node','recharge_rate')
recharge[:] = R
grid.at_node["surface_water__discharge"] = grid.add_zeros(
            "node", "effective_drainage_area")

#initialize components
fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8',  \
                      runoff_rate='recharge_rate')
lmb = LakeMapperBarnes(grid, method='D8', fill_flat=False,
                              surface='topographic__elevation',
                              fill_surface='topographic__elevation',
                              redirect_flow_steepest_descent=False,
                              reaccumulate_flow=False,
                              track_lakes=False,
                              ignore_overfill=True)
sp = FastscapeEroder(grid,K_sp = K,m_sp = m, n_sp=n,discharge_field='surface_water__discharge')
ld = LinearDiffuser(grid, linear_diffusivity=D)
dfr = DepressionFinderAndRouter(grid)

#run model forward
for i in range(N):

    dfr._find_pits()
    if dfr._number_of_pits > 0:
        lmb.run_one_step()

    fa.run_one_step()

    area = grid.at_node["drainage_area"]
    slope = grid.at_node["topographic__steepest_slope"]
    sat_param = Ks * grid.dx * (elev-base) / recharge
    eff_area = area * ( np.exp(-sat_param * slope / area) )
    grid.at_node["surface_water__discharge"] = eff_area

    grid.at_node['topographic__elevation'][cores] += uplift_rate*dt
    grid.at_node['aquifer_base__elevation'][cores] += uplift_rate*dt - w0*np.exp(-(elev[grid.core_nodes]-base[grid.core_nodes])/d_s)*dt

    sp.run_one_step(dt)
    ld.run_one_step(dt)
