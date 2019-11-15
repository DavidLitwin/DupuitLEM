# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:56:40 2019

@author: dgbli
"""


import numpy as np

from landlab import RasterModelGrid, FIXED_VALUE_BOUNDARY, CLOSED_BOUNDARY
from landlab.components import (
    FlowAccumulator,
    FastscapeEroder,
    LinearDiffuser,
    SinkFillerBarnes,
    )
from landlab.io.netcdf import write_raster_netcdf
from landlab.utils import return_array_at_node

# Set parameters
R = 1.5/(365*24*3600)  # steady, uniform recharge rate [m/s]
uplift_rate = 1E-4/(365*24*3600) # uniform uplift [m/s]
m = 0.5 #Exponent on Q []
n = 1.0 #Exponent on S []
K = 5E-8 #erosivity coefficient [m-1/2 s−1/2]
D = 0.01/(365*24*3600) # hillslope diffusivity [m2/s]

dt_h = 1E5 # hydrological timestep [s]
T = 1e6*(365*24*3600) # total simulation time [s]
MSF = 5000 # morphologic scaling factor [-]
dt_m = MSF*dt_h
N = T//dt_m
N = int(N)
output_interval = 2500

# Set output options
output_fields = [
            "topographic__elevation",
            ]
time_unit="years",
reference_time="model start",
space_unit="meters",

# Set boundary and ititial conditions
np.random.seed(2)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=CLOSED_BOUNDARY, top=CLOSED_BOUNDARY, \
                              left=FIXED_VALUE_BOUNDARY, bottom=CLOSED_BOUNDARY)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = 0.1*np.random.rand(len(elev))
precip = grid.add_zeros("node", "precipitation__rate")
precip[:] = return_array_at_node(grid,R)

sf = SinkFillerBarnes(grid, method='D8')
sf.run_one_step()

fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8',  \
                     depression_finder = 'DepressionFinderAndRouter', runoff_rate='precipitation__rate')
sp = FastscapeEroder(grid,K_sp = K,m_sp = m, n_sp=n, discharge_name='surface_water__discharge')
ld = LinearDiffuser(grid, linear_diffusivity=D)

# Run model forward
num_substeps = np.zeros(N)
max_rel_change = np.zeros(N)
perc90_rel_change = np.zeros(N)
for i in range(N):
    elev0 = elev.copy()

    fa.run_one_step()

    grid.at_node['topographic__elevation'][grid.core_nodes] += uplift_rate*dt_m

    sp.run_one_step(dt_m)
    ld.run_one_step(dt_m)

    if i % output_interval == 0:

        filename = './data/grid_' + str(i) + '.nc'
        write_raster_netcdf(
                filename, grid, names=output_fields, format="NETCDF4")
        print('Completed loop %d' % i)

    elev_diff = abs(elev-elev0)/elev0
    max_rel_change[i] = np.max(elev_diff)
    perc90_rel_change[i] = np.percentile(elev_diff,90)

    if perc90_rel_change[i] < 1e-5:
        break


filename = './data/grid_' + str(i) + '.nc'
write_raster_netcdf(filename, grid, names=output_fields, format="NETCDF4")

filename = './data/max_rel_change' + '.txt'
np.savetxt(filename,max_rel_change)

filename = './data/90perc_rel_change' + '.txt'
np.savetxt(filename,perc90_rel_change)
