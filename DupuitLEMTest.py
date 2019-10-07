# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:49:06 2019

@author: dgbli
"""

import time
import numpy as np
import pickle

from landlab import RasterModelGrid, FIXED_VALUE_BOUNDARY, CLOSED_BOUNDARY
from landlab.components import GroundwaterDupuitPercolator, FlowAccumulator, FastscapeEroder, LinearDiffuser

#%% Parameters

R = 5e-8  # steady, uniform recharge rate, in m/s
Kiso = 0.0001  # uniform hydraulic conductivity, in m/s

uplift_rate = 1E-4/(365*24*3600) #m/s
m = 0.5 #Exponent on A []
n = 1.0 #Exponent on S []
K = 5E-9 #erosivity coefficient [s−1]
D = 0.005/(365*24*3600) #m2/s

w0 = 2E-4/(365*24*3600) #max rate of soil production
dc = 2 #m characteristic soil depth

#%% Initialize grid and components

# set boundary and ititial conditions
grid = RasterModelGrid((21, 41), spacing=10.0)
grid.set_status_at_node_on_edges(right=CLOSED_BOUNDARY, top=CLOSED_BOUNDARY, \
                              left=FIXED_VALUE_BOUNDARY, bottom=CLOSED_BOUNDARY)
elev = grid.add_zeros('node', 'topographic__elevation')
np.random.seed(2)
elev[:] = grid.x_of_node/100+2  + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
base[:] = grid.x_of_node/100
wt = grid.add_zeros('node', 'water_table__elevation')

#left boundary has water table elevation fixed at surface elevation
wt[:] = grid.x_of_node/100+2

# initialize the groundwater model
gdp = GroundwaterDupuitPercolator(grid, porosity=0.2, hydraulic_conductivity=Kiso, \
                                  recharge_rate=R,regularization_f=0.01)
fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8',  \
                     depression_finder = 'DepressionFinderAndRouter', runoff_rate='surface_water__specific_discharge')
sp = FastscapeEroder(grid,K_sp = K,m_sp = m, n_sp=n,discharge_name='surface_water__discharge')
ld = LinearDiffuser(grid, linear_diffusivity=D)

#%% Run model forward
t0 = time.time()

N = 100000
dt_h = 1E5
dt_m = 500*dt_h
num_substeps = np.zeros(N)

sw_flux = np.zeros_like(elev)
gw_flux = np.zeros_like(elev)

for i in range(N):
    
    gdp.run_with_adaptive_time_step_solver(dt_h,courant_coefficient=0.02)
    
    num_substeps[i] = gdp._num_substeps

    #debug
    if np.isnan(gdp._thickness).any():
        break

    fa.run_one_step()

    gw_flux = gdp.calc_gw_flux_at_node()
    sw_flux = grid.at_node['surface_water__discharge']


    grid.at_node['topographic__elevation'][grid.core_nodes] += uplift_rate*dt_m
    grid.at_node['aquifer_base__elevation'][grid.core_nodes] += uplift_rate*dt_m - w0*np.exp(-(elev[grid.core_nodes]-base[grid.core_nodes])/dc)*dt_m
    
    sp.run_one_step(dt_m)
    ld.run_one_step(dt_m)
    
    elev[elev<base] = base[elev<base]

    if i % 100 == 0:
        print ('Completed loop %d' % i)

t1 = time.time()

tot_time = t1-t0

#%% Pickle results

pickle.dump(grid, open('./Dupuit_LEM_results/grid_'+str(time.time())+'.p','wb'))
    