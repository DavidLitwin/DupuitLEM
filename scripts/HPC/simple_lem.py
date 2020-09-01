# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:11:59 2020

A simple streampower diffusion model, to test the length scales in the
nondimensionalization introduced by Nikos Theodoratos.

@author: dgbli
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
from landlab.io.netcdf import write_raster_netcdf


task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_path = './data/simple_lem_1_'

dx = 16.0
lg_all = dx/np.array([0.9, 0.8, 0.75, 0.7, 0.6])

lg = lg_all[ID]

D = 0.01 #m2/yr
K = D/lg**2
U = 1e-4 #m/yr
m = 0.5
n = 1

hg = U/K
T = 600*(1/K)
dt = 5e-3*(1/K)
N = int(T//dt)
output_interval = 1000

np.random.seed(12345)
grid = RasterModelGrid((125,125), xy_spacing=dx)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED,
                                top=grid.BC_NODE_IS_CLOSED,
                                left=grid.BC_NODE_IS_FIXED_VALUE,
                                bottom=grid.BC_NODE_IS_CLOSED,
)

z = grid.add_zeros('node', 'topographic__elevation')
z[:] = 0.01*hg*np.random.rand(len(z))


fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8', depression_finder='LakeMapperBarnes')
ld = LinearDiffuser(grid, D)
sp = FastscapeEroder(grid, K_sp=K, m_sp=m, n_sp=n)

times = np.zeros(int(N//output_interval))
t1 = time.time()
for i in range(N):

    z[grid.core_nodes] += U*dt

    ld.run_one_step(dt)
    fa.run_one_step()
    sp.run_one_step(dt)

    if i%output_interval==0:
        print('finished iteration %d'%i)
        filename = base_path + '%d_grid_%d.nc'%(ID,i)
        write_raster_netcdf(filename, grid, names = "topographic__elevation", format="NETCDF4")

        t = time.time()
        times[i//output_interval] = t-t1
        t1 = t

np.savetxt('times.out', times)
