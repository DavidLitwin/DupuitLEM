# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:11:59 2020

@author: dgbli
"""

import os
import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    LinearDiffuser,
    FlowAccumulator,
    FastscapeEroder,
    PrecipitationDistribution,
    )
from landlab.io.netcdf import write_raster_netcdf


task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_path = './data/simple_lem_1_'

lg_all = [40, 80, 160, 320, 640]
lg = lg_all[ID]

D = 6e-11
K = D/lg**2
U = 3e-12
m = 0.5
n = 1

T = 5e6*(365*24*3600)
dt = 50*(365*24*3600)
N = int(T//dt)

grid = RasterModelGrid((100,100), xy_spacing=lg/4)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
z = grid.add_zeros('node', 'topographic__elevation')
z[:] = 0.1*np.random.rand(len(z))


fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8', depression_finder='LakeMapperBarnes')
ld = LinearDiffuser(grid, D)
sp = FastscapeEroder(grid, K_sp=K, m_sp=m, n_sp=n)

for i in range(N):

    z[grid.core_nodes] += U*dt

    ld.run_one_step(dt)
    fa.run_one_step()
    sp.run_one_step(dt)


    if i%1000==0:
        print('finished iteration %d'%i)
        filename = base_path + '%d_grid_%d.nc'%(ID,i)
        write_raster_netcdf(filename, grid, names = "topographic__elevation", format="NETCDF4")
