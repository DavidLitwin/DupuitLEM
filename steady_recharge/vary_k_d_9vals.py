# -*- coding: utf-8 -*-
"""
Created on 19 Nov 2019

Test three values of ksat: 0.2, 1, 2 m/hr, and three values of d_s: 0.5, 1.0, 2.0

@author: dgbli
"""
import os
import time
import numpy as np
from itertools import product

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    FlowAccumulator,
    FastscapeEroder,
    LinearDiffuser,
    LakeMapperBarnes,
    DepressionFinderAndRouter,
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
Ks_all = np.array([0.2, 1, 2])/(3600)  # hydraulic conductivity at the surface [m/s]
w0 = 2E-4/(365*24*3600) #max rate of soil production [m/s]
d_i_rel = 1.0 # initial depth relative to steady state depth [-]
d_s_all = [0.5,1.0,2.0] # characteristic soil production depth [m]
uplift_rate = 1E-4/(365*24*3600) # uniform uplift [m/s]
m = 0.5 #Exponent on Q []
n = 1.0 #Exponent on S []
K = 5E-8 #erosivity coefficient [m-1/2 s−1/2]
D = 0.01/(365*24*3600) # hillslope diffusivity [m2/s]

params = np.array(list(product(Ks_all,d_s_all)))

id = int(task_id)
Ks = params[id,0] # hydraulic conductivity
K0 = 0.01*Ks # asymptotic hydraulic conductivity at infinite depth
d_s = params[id,1] # characteristic soil production depth [m]
d_k = d_s # characteristic depth for hydraulic conductivity [m]
d_i = d_i_rel*(-d_s*np.log(uplift_rate/w0)) # initial permeable thickness

dt_h = 1E5 # hydrological timestep [s]
T = 5e5*(365*24*3600) # total simulation time [s]
MSF = 1000 # morphologic scaling factor [-]
dt_m = MSF*dt_h
N = T//dt_m
N = int(N)
output_interval = 5000

# Set output options
output_fields = [
            "topographic__elevation",
            "aquifer_base__elevation",
            "water_table__elevation",
            "surface_water__discharge",
            "storm_average_surface_water__specific_discharge",
            "groundwater__specific_discharge_node"
            ]
time_unit="years",
reference_time="model start",
space_unit="meters",

# Set boundary and ititial conditions
np.random.seed(2)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = d_i + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()
gw_flux = grid.add_zeros('node', 'groundwater__specific_discharge_node')
Kavg = avg_hydraulic_conductivity(grid,wt-base,elev-base,K0,Ks,d_k ) # depth-averaged hydraulic conductivity

# initialize model components
gdp = GroundwaterDupuitPercolator(grid, porosity=0.2, hydraulic_conductivity=Kavg, \
                                  recharge_rate=R,regularization_f=0.01, courant_coefficient=0.2)
fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8',  \
                      runoff_rate='average_surface_water__specific_discharge')
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

# Run model forward
num_substeps = np.zeros(N)
max_rel_change = np.zeros(N)
perc90_rel_change = np.zeros(N)
times = np.zeros((N,6))
num_pits = np.zeros(N)
t0 = time.time()
for i in range(N):
    elev0 = elev.copy()

    t1 = time.time()
    #set hydraulic conductivity based on depth
    gdp.K = avg_hydraulic_conductivity(grid,grid.at_node['aquifer__thickness'],
                                     grid.at_node['topographic__elevation']-
                                     grid.at_node['aquifer_base__elevation'],
                                     K0,Ks,d_k,
                                     )
    #run gw model
    gdp.run_with_adaptive_time_step_solver(dt_h)
    num_substeps[i] = gdp.number_of_substeps

    t2 = time.time()

    #uplift and regolith production
    grid.at_node['topographic__elevation'][grid.core_nodes] += uplift_rate*dt_m
    grid.at_node['aquifer_base__elevation'][grid.core_nodes] += uplift_rate*dt_m - w0*np.exp(-(elev[grid.core_nodes]-base[grid.core_nodes])/d_s)*dt_m

    t3 = time.time()
    dfr._find_pits()

    t4 = time.time()
    if dfr._number_of_pits > 0:
        lmb.run_one_step()

    t5 = time.time()
    fa.run_one_step()

    t6 = time.time()
    ld.run_one_step(dt_m)
    sp.run_one_step(dt_m)

    elev[elev<base] = base[elev<base]

    t7 = time.time()

    if i%100 == 0:
        num_pits[i//100] = dfr._number_of_pits
        times[i//100,:] = [t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6]

    ############# record output ##############

    if i % output_interval == 0:
        gw_flux[:] = gdp.calc_gw_flux_at_node()

        filename = './data/vary_k_d_' + str(task_id) + '_grid_' + str(i) + '.nc'
        write_raster_netcdf(
                filename, grid, names=output_fields, format="NETCDF4")
        print('Completed loop %d' % i)

        filename = './data/vary_k_d_' + str(task_id) + '_substeps' + '.txt'
        np.savetxt(filename,num_substeps, fmt='%.1f')

        filename = './data/vary_k_d_' + str(task_id) + '_max_rel_change' + '.txt'
        np.savetxt(filename,max_rel_change, fmt='%.4e')

        filename = './data/vary_k_d_' + str(task_id) + '_90perc_rel_change' + '.txt'
        np.savetxt(filename,perc90_rel_change, fmt='%.4e')

        filename = './data/vary_k_d_' + str(task_id) + '_num_pits' + '.txt'
        np.savetxt(filename,num_pits, fmt='%.1f')

        filename = './data/vary_k_d_' + str(task_id) + '_times' + '.txt'
        np.savetxt(filename,times, fmt='%.4e')

    elev_diff = abs(elev-elev0)/elev0
    max_rel_change[i] = np.max(elev_diff)
    perc90_rel_change[i] = np.percentile(elev_diff,90)

    # if perc90_rel_change[i] < 1e-6:
    #     break

tfin = time.time()
tot_time = tfin-t0
# df_times = pd.DataFrame(times,columns=['gdp_s', 'fa_s', 'gdp_i', 'fa_i', 'up', 'sp', 'ld'])
# filename = './data/' + job_id + '_' + task_id + '_times.p'
# pickle.dump(df_times,open(filename,'wb'))

# collect output and save
gw_flux[:] = gdp.calc_gw_flux_at_node()

filename = './data/vary_k_d_' + str(task_id) + '_grid_' + str(i) + '.nc'
write_raster_netcdf(filename, grid, names=output_fields, format="NETCDF4")

filename = './data/vary_k_d_' + str(task_id) + '_time' + '.txt'
timefile = open(filename,'w')
timefile.write('Run time: ' + str(tot_time))
timefile.close()

filename = './data/vary_k_d_' + str(task_id) + '_substeps' + '.txt'
np.savetxt(filename,num_substeps)

filename = './data/vary_k_d_' + str(task_id) + '_max_rel_change' + '.txt'
np.savetxt(filename,max_rel_change)

filename = './data/vary_k_d_' + str(task_id) + '_90perc_rel_change' + '.txt'
np.savetxt(filename,perc90_rel_change)
