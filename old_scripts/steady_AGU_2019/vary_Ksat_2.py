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
from landlab.grid.mappers import map_mean_of_link_nodes_to_link

def avg_hydraulic_conductivity(grid,h,b,k0,ks,dk):
    """
    Calculate the average hydraulic conductivity when hydraulic conductivity
    varies with depth as:

        k = k0 + (ks-k0)*exp(-d/dk)

    Parameters:
        h = aquifer thickness
        b = depth from surface to impermeable base
        k0: asymptotic permeability at infinite depth
        ks: hydraulic conductivity at the ground surface
        dk = characteristic depth

    """

    blink = map_mean_of_link_nodes_to_link(grid,b)
    hlink = map_mean_of_link_nodes_to_link(grid,h)
    b1 = blink[hlink>0.0]
    h1 = hlink[hlink>0.0]
    kavg = np.zeros_like(hlink)
    kavg[hlink>0.0] = dk*(ks-k0)/h1 * (np.exp(-(b1-h1)/dk) - np.exp(-b1/dk)) + k0
    return kavg

task_id = os.environ['SLURM_ARRAY_TASK_ID']
job_id = os.environ['SLURM_ARRAY_JOB_ID']

# Set parameters
R = 1.5/(365*24*3600)  # steady, uniform recharge rate [m/s]
Ks_all = np.linspace(2.5,4.8,24)/(3600)  # hydraulic conductivity at the surface [m/s]
Ks_print_all = (Ks_all*3600).astype(str)
w0 = 2E-4/(365*24*3600) #max rate of soil production [m/s]
d_k = 1.5 # characteristic depth for hydraulic conductivity [m]
d_s = 1.5 # characteristic soil production depth [m]
uplift_rate = 1E-4/(365*24*3600) # uniform uplift [m/s]
di = 1*(-d_s*np.log(uplift_rate/w0)) # initial depth relative to steady state depth [-]
m = 0.5 #Exponent on Q []
n = 1.0 #Exponent on S []
K = 5E-8 #erosivity coefficient [m-1/2 s−1/2]
D = 0.01/(365*24*3600) # hillslope diffusivity [m2/s]

ID = int(task_id)
Ks_print = Ks_print_all[ID]
Ks = Ks_all[ID] # hydraulic conductivity
K0 = 0.01*Ks # asymptotic hydraulic conductivity at infinite depth

dt_h = 1E5 # hydrological timestep [s]
T = 5e5*(365*24*3600) # total simulation time [s]
MSF = 500 # morphologic scaling factor [-]
dt_m = MSF*dt_h
N = T//dt_m
N = int(N)
output_interval = 2500

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
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=CLOSED_BOUNDARY, top=CLOSED_BOUNDARY, \
                              left=FIXED_VALUE_BOUNDARY, bottom=CLOSED_BOUNDARY)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = di + 0.1*np.random.rand(len(elev))

sf = SinkFillerBarnes(grid, method='D8')
sf.run_one_step()

base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev
gw_flux = grid.add_zeros('node', 'groundwater__specific_discharge_node')
Kavg = avg_hydraulic_conductivity(grid,wt-base,elev-base,K0,Ks,d_k ) # depth-averaged hydraulic conductivity

# initialize model components
gdp = GroundwaterDupuitPercolator(grid, porosity=0.2, hydraulic_conductivity=Kavg, \
                                  recharge_rate=R,regularization_f=0.01)
fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8',  \
                     depression_finder = 'DepressionFinderAndRouter', runoff_rate='average_surface_water__specific_discharge')
sp = FastscapeEroder(grid,K_sp = K,m_sp = m, n_sp=n,discharge_name='surface_water__discharge')
ld = LinearDiffuser(grid, linear_diffusivity=D)

# Run model forward
num_substeps = np.zeros(N)
max_rel_change = np.zeros(N)
perc90_rel_change = np.zeros(N)
t0 = time.time()
for i in range(N):
    elev0 = elev.copy()

    #set hydraulic conductivity based on depth
    gdp.K = avg_hydraulic_conductivity(grid,grid.at_node['aquifer__thickness'],
                                     grid.at_node['topographic__elevation']-
                                     grid.at_node['aquifer_base__elevation'],
                                     K0,Ks,d_k,
                                     )
    #run gw model
    gdp.run_with_adaptive_time_step_solver(dt_h,courant_coefficient=0.2)
    num_substeps[i] = gdp._num_substeps

    fa.run_one_step()

    grid.at_node['topographic__elevation'][grid.core_nodes] += uplift_rate*dt_m
    grid.at_node['aquifer_base__elevation'][grid.core_nodes] += uplift_rate*dt_m - w0*np.exp(-(elev[grid.core_nodes]-base[grid.core_nodes])/d_s)*dt_m

    sp.run_one_step(dt_m)
    ld.run_one_step(dt_m)

    elev[elev<base] = base[elev<base]

    if i % output_interval == 0:
        gw_flux[:] = gdp.calc_gw_flux_at_node()

        filename = './data/vary_Ksat_' + Ks_print + '_grid_' + str(i) + '.nc'
        write_raster_netcdf(
                filename, grid, names=output_fields, format="NETCDF4")
        print('Completed loop %d' % i)

        filename = './data/vary_Ksat_' + Ks_print + '_substeps' + '.txt'
        np.savetxt(filename,num_substeps)

        filename = './data/vary_Ksat_' + Ks_print + '_max_rel_change' + '.txt'
        np.savetxt(filename,max_rel_change)

        filename = './data/vary_Ksat_' + Ks_print + '_90perc_rel_change' + '.txt'
        np.savetxt(filename,perc90_rel_change)

    elev_diff = abs(elev-elev0)/elev0
    max_rel_change[i] = np.max(elev_diff)
    perc90_rel_change[i] = np.percentile(elev_diff,90)

    if perc90_rel_change[i] < 1e-6:
        break


tfin = time.time()
tot_time = tfin-t0

# collect output and save
gw_flux[:] = gdp.calc_gw_flux_at_node()

filename = './data/vary_Ksat_' + Ks_print + '_grid_' + str(i) + '.nc'
write_raster_netcdf(filename, grid, names=output_fields, format="NETCDF4")

filename = './data/vary_Ksat_' + Ks_print + '_time' + '.txt'
timefile = open(filename,'w')
timefile.write('Run time: ' + str(tot_time))
timefile.close()

filename = './data/vary_Ksat_' + Ks_print + '_substeps' + '.txt'
np.savetxt(filename,num_substeps)

filename = './data/vary_Ksat_' + Ks_print + '_max_rel_change' + '.txt'
np.savetxt(filename,max_rel_change)

filename = './data/vary_Ksat_' + Ks_print + '_90perc_rel_change' + '.txt'
np.savetxt(filename,perc90_rel_change)
