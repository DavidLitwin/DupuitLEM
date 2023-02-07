# -*- coding: utf-8 -*-
"""
NoHyd model Q*=1 for all (x,y).

This script uses characteristic scales and dimensionless parameters presented
in Litwin et al. 2021. Vary lg and hg to confirm the independence of these
length scales.

Date: 2 Jan 2020
"""

import os
import numpy as np
import pandas
import pickle
from itertools import product

from landlab import RasterModelGrid
from landlab.components import (
    LinearDiffuser,
    FlowAccumulator,
    FastscapeEroder,
    )
from landlab.io.netcdf import to_netcdf

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_path = './data/simple_lem_K_D_U_'

K_all = [1.144342e-13, 7.425230e-14] # DR, BR
D = 5.327475e-10 # m2/s
U = 3.623215e-13 # m/s
m_sp = 0.5
n_sp = 1.0
v0 = 10 # m 

Ksp = K_all[ID]

T = 5e6 * 3600 * 24 * 365
dt = 50 * 3600 * 24 * 365

N = int(T//dt)
output_interval = 1000

np.random.seed(12345)
grid = RasterModelGrid((125,125), xy_spacing=v0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED,
                                top=grid.BC_NODE_IS_CLOSED,
                                left=grid.BC_NODE_IS_FIXED_VALUE,
                                bottom=grid.BC_NODE_IS_CLOSED,
)
z = grid.add_zeros('node', 'topographic__elevation')
z[:] = 0.1*np.random.rand(len(z))

fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8', depression_finder='LakeMapperBarnes')
ld = LinearDiffuser(grid, D)
sp = FastscapeEroder(grid, K_sp=Ksp, m_sp=m_sp, n_sp=n_sp)

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
