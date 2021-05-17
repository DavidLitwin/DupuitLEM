# -*- coding: utf-8 -*-
"""
A simple streampower diffusion model, to test the grid scale dependence of
the solution with nondimensionalization based on Theodoratos/Bonetti.

Date: 9 Jan 2021
"""

import os
import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    LinearDiffuser,
    FlowAccumulator,
    FastscapeEroder,
    )
from landlab.io.netcdf import to_netcdf

def K_fun(a0, lg, tg):
    return np.sqrt(lg)/(np.sqrt(a0)*tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_path = './data/simple_lem_4_'

lg = 15
hg = 2.25 # geomorphic height scale [m]
tg = 22500 # geomorphic timescale [yr]
a0 = 15 #valley width factor [m]
m = 0.5
n = 1

Lx_nd = 87.5
v0_all = lg*np.array([1.5, 1.3, 1.1, 1, 0.9, 0.7, 0.5])
Nx_all = (Lx_nd*lg+1e-10)//v0_all
v0 = v0_all[ID]
Nx = int(Nx_all[ID])

D = D_fun(lg, tg)
K = K_fun(a0, lg, tg)
Ksp = K*(a0/v0)**m # corrected for valley width
U = U_fun(hg, tg)

T = 2000*tg
dt = 2e-3*tg
N = int(T//dt)
output_interval = 2000

np.random.seed(12345)
grid = RasterModelGrid((Nx, Nx), xy_spacing=v0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
z = grid.add_zeros('node', 'topographic__elevation')
z[:] = 0.1*hg*np.random.rand(len(z))


fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8', depression_finder='LakeMapperBarnes')
ld = LinearDiffuser(grid, D)
sp = FastscapeEroder(grid, K_sp=Ksp, m_sp=m, n_sp=n, discharge_field='drainage_area')

for i in range(N):

    z[grid.core_nodes] += U*dt

    ld.run_one_step(dt)
    fa.run_one_step()
    sp.run_one_step(dt)

    if i%output_interval==0:
        print('finished iteration %d'%i)
        filename = base_path + '%d_grid_%d.nc'%(ID,i)
        to_netcdf(grid, filename, include="at_node:topographic__elevation")
