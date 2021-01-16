# -*- coding: utf-8 -*-
"""
NoHyd model Q*=1 for all (x,y).
Vary lg and hg to confirm the independence of these length scales
in this model case.

Date: 2 Jan 2020
"""

import os
import time
import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    LinearDiffuser,
    FlowAccumulator,
    FastscapeEroder,
    )
from landlab.io.netcdf import to_netcdf

#dim equations
def K_fun(a0, lg, tg):
    return np.sqrt(lg)/(np.sqrt(a0)*tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_path = './data/simple_lem_5_'

lg_all = np.array([15, 15, 15]) # geomorphic length scale [m]
hg_all = np.array([2.25, 4.5, 9]) # geomorphic height scale [m]
lg = lg_all[ID] # geomorphic length scale [m]
hg = hg_all[ID] # geomorphic height scale [m]
tg = 22500 # geomorphic timescale [yr]
a0 = 0.7*15 #valley width factor [m]
v0 = 0.7*lg #grid cell width [m]

D = D_fun(lg, tg)
K = K_fun(a0, lg, tg)
Ksp = K*np.sqrt(a0/v0)
U = U_fun(hg, tg)
m = 0.5
n = 1

T = 2000*tg
dt = 2e-3*tg
N = int(T//dt)
output_interval = 2000

np.random.seed(12345)
grid = RasterModelGrid((125,125), xy_spacing=v0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED,
                                top=grid.BC_NODE_IS_CLOSED,
                                left=grid.BC_NODE_IS_FIXED_VALUE,
                                bottom=grid.BC_NODE_IS_CLOSED,
)
z = grid.add_zeros('node', 'topographic__elevation')
z[:] = 0.1*hg*np.random.rand(len(z))


fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8', depression_finder='LakeMapperBarnes')
ld = LinearDiffuser(grid, D)
sp = FastscapeEroder(grid, K_sp=Ksp, m_sp=m, n_sp=n)

for i in range(N):

    z[grid.core_nodes] += U*dt

    ld.run_one_step(dt)
    fa.run_one_step()
    sp.run_one_step(dt)

    # print('completed loop %d'%i)

    if i%output_interval==0:
        print('finished iteration %d'%i)
        filename = base_path + '%d_grid_%d.nc'%(ID,i)
        to_netcdf(grid, filename, include="at_node:topographic__elevation")
