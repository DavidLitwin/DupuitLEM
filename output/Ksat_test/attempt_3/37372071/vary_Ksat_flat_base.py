# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:49:06 2019

@author: dgbli
"""
import os
import time
import numpy as np

from landlab import RasterModelGrid, FIXED_VALUE_BOUNDARY, CLOSED_BOUNDARY
from landlab.components import (
    GroundwaterDupuitPercolator,
    FlowAccumulator,
    FastscapeEroder,
    LinearDiffuser,
    SinkFillerBarnes,
    )
from landlab.io.netcdf import write_raster_netcdf

task_id = os.environ['SLURM_ARRAY_TASK_ID']
job_id = os.environ['SLURM_ARRAY_JOB_ID']

# Set parameters
R = 1.5/(365*24*3600)  # steady, uniform recharge rate, in m/s
Kiso_all = np.array([5e-5, 1e-4, 5e-4])  # uniform hydraulic conductivity, in m/s
Kiso = Kiso_all[int(task_id)]
uplift_rate = 1E-4/(365*24*3600) #m/s
m = 0.5 #Exponent on A []
n = 1.0 #Exponent on S []
K = 5E-8 #erosivity coefficient [m-1/2 s−1/2]
D = 0.01/(365*24*3600) #m2/s
w0 = 5E-4/(365*24*3600) #max rate of soil production
dc = 2 #m characteristic soil depth
dt_h = 1E5
T = 250000*(365*24*3600)
MSF = 5000
dt_m = MSF*dt_h
N = T//dt_m
N = int(N)
output_interval = 250

# Set output options
output_fields = [
            "topographic__elevation",
            "aquifer_base__elevation",
            "water_table__elevation",
            "surface_water__discharge",
            "groundwater__specific_discharge_node"
            ]
time_unit="years",
reference_time="model start",
space_unit="meters",

# Set boundary and ititial conditions
np.random.seed(2)
grid = RasterModelGrid((100, 100), spacing=10.0)
grid.set_status_at_node_on_edges(right=CLOSED_BOUNDARY, top=CLOSED_BOUNDARY, \
                              left=FIXED_VALUE_BOUNDARY, bottom=CLOSED_BOUNDARY)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = 2 + 0.1*np.random.rand(len(elev))

sf = SinkFillerBarnes(grid, method='D8')
sf.run_one_step()

base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev
gw_flux = grid.add_zeros('node', 'groundwater__specific_discharge_node')

# initialize model components
gdp = GroundwaterDupuitPercolator(grid, porosity=0.2, hydraulic_conductivity=Kiso, \
                                  recharge_rate=R,regularization_f=0.01)
fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8',  \
                     depression_finder = 'DepressionFinderAndRouter', runoff_rate='surface_water__specific_discharge')
sp = FastscapeEroder(grid,K_sp = K,m_sp = m, n_sp=n,discharge_name='surface_water__discharge')
ld = LinearDiffuser(grid, linear_diffusivity=D)

# Run model forward
num_substeps = np.zeros(N)
max_rel_change = np.zeros(N)
t0 = time.time()
for i in range(N):
    elev0 = elev.copy()

    gdp.run_with_adaptive_time_step_solver(dt_h,courant_coefficient=0.02)
    num_substeps[i] = gdp._num_substeps

    #for later debugging
    if np.isnan(gdp._thickness).any():
        print('NaN in thickness')
        gw_flux[:] = gdp.calc_gw_flux_at_node()
        filename = './data/' + job_id + '_' + str(Kiso) + '_grid_at_nan_' + str(i) + '.nc'
        write_raster_netcdf(
                filename, grid, names=output_fields, format="NETCDF4")
        break

    if np.isinf(gdp._thickness).any():
        print('Inf in thickness')
        gw_flux[:] = gdp.calc_gw_flux_at_node()
        filename = './data/' + job_id + '_' + str(Kiso) + '_grid_at_inf_' + str(i) + '.nc'
        write_raster_netcdf(
                filename, grid, names=output_fields, format="NETCDF4")
        break

    fa.run_one_step()

    grid.at_node['topographic__elevation'][grid.core_nodes] += uplift_rate*dt_m
    grid.at_node['aquifer_base__elevation'][grid.core_nodes] += uplift_rate*dt_m - w0*np.exp(-(elev[grid.core_nodes]-base[grid.core_nodes])/dc)*dt_m

    sp.run_one_step(dt_m)
    ld.run_one_step(dt_m)

    elev[elev<base] = base[elev<base]

    if i % output_interval == 0:
        gw_flux[:] = gdp.calc_gw_flux_at_node()

        filename = './data/' + job_id + '_' + str(Kiso) + '_grid_' + str(i) + '.nc'
        write_raster_netcdf(
                filename, grid, names=output_fields, format="NETCDF4")
        print('Completed loop %d' % i)

    max_rel_change[i] = np.max(abs(elev-elev0)/elev0)

t1 = time.time()

tot_time = t1-t0

# collect output and save
gw_flux[:] = gdp.calc_gw_flux_at_node()

filename = './data/' + job_id + '_' + str(Kiso) + '_grid_end' + '.nc'
write_raster_netcdf(filename, grid, names=output_fields, format="NETCDF4")

filename = './data/' + job_id + '_' + str(Kiso) + '_time' + '.txt'
timefile = open(filename,'w')
timefile.write('Run time: ' + str(tot_time))
timefile.close()

filename = './data/' + job_id + '_' + str(Kiso) + '_substeps' + '.txt'
np.savetxt(filename,num_substeps)

filename = './data/' + job_id + '_' + str(Kiso) + '_max_rel_change' + '.txt'
np.savetxt(filename,max_rel_change)
