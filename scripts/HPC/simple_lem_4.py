# -*- coding: utf-8 -*-
"""

A simple streampower diffusion model, to test the grid scale dependence of
the solution with lg, the geomorphic length scale defined by Theodoratos.

Implement a correction factor for grid scale dependence

@author: dgbli
20 Nov 2020

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


task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_path = './data/simple_lem_4_'

lg = 15
Lx_nd = 87.5
dx_all = lg*np.array([2, 1.5, 1.25, 1, 0.75, 0.5])
Nx_all = (Lx_nd*lg+1e-10)//dx_all

dx = dx_all[ID]
Nx = int(Nx_all[ID])

w0 = lg
A0 = lg**2

D = 0.01 #m2/yr
K = (D/lg**2) #*(dx/w0)
U = 1e-4 #m/yr
m = 0.5
n = 1

hg = U/K
T = 800*(1/K)
dt = 5e-3*(1/K)
N = int(T//dt)
output_interval = 1000

np.random.seed(12345)
grid = RasterModelGrid((Nx, Nx), xy_spacing=dx)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
z = grid.add_zeros('node', 'topographic__elevation')
z[:] = 0.1*hg*np.random.rand(len(z))


fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8', depression_finder='LakeMapperBarnes')
ld = LinearDiffuser(grid, D)

accum = grid.add_zeros('node', 'width_norm_area')
sp = FastscapeEroder(grid, K_sp=K, m_sp=m, n_sp=n, discharge_field='width_norm_area')

for i in range(N):

    z[grid.core_nodes] += U*dt

    ld.run_one_step(dt)
    fa.run_one_step()
    accum[:] = grid.at_node['drainage_area']*(A0/dx**2)
    sp.run_one_step(dt)

    print('completed loop %d'%i)

    if i%output_interval==0:
        print('finished iteration %d'%i)
        filename = base_path + '%d_grid_%d.nc'%(ID,i)
        to_netcdf(grid, filename, include="at_node:topographic__elevation")
