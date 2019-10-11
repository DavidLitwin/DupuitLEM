# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:49:06 2019

@author: dgbli
"""
import os
import time
import numpy as np
import xarray as xr
import pickle

from landlab import RasterModelGrid, FIXED_VALUE_BOUNDARY, CLOSED_BOUNDARY
from landlab.components import GroundwaterDupuitPercolator, FlowAccumulator, FastscapeEroder, LinearDiffuser
from landlab.io.netcdf import write_raster_netcdf

# Set parameters
R = 1.5/(365*24*3600)  # steady, uniform recharge rate, in m/s
Kiso = 0.0001  # uniform hydraulic conductivity, in m/s
uplift_rate = 1E-4/(365*24*3600) #m/s
m = 0.5 #Exponent on A []
n = 1.0 #Exponent on S []
K = 5E-8 #erosivity coefficient [m-1/2 s−1/2]
D = 0.005/(365*24*3600) #m2/s
w0 = 2E-3/(365*24*3600) #max rate of soil production
dc = 2 #m characteristic soil depth
N = 50000
dt_h = 1E5
dt_m = 500*dt_h

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
output_interval = 1000

# Set boundary and ititial conditions
np.random.seed(2)
grid = RasterModelGrid((100, 100), spacing=10.0)
grid.set_status_at_node_on_edges(right=CLOSED_BOUNDARY, top=CLOSED_BOUNDARY, \
                              left=FIXED_VALUE_BOUNDARY, bottom=CLOSED_BOUNDARY)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = grid.x_of_node/500+2  + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
base[:] = grid.x_of_node/500
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = grid.x_of_node/500+2
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
filenames = []
time_list = []
t0 = time.time()
for i in range(N):

    gdp.run_with_adaptive_time_step_solver(dt_h,courant_coefficient=0.02)
    num_substeps[i] = gdp._num_substeps

    #debug
    if np.isnan(gdp._thickness).any():
        print('NaN in thickness')
        break
    if np.isinf(gdp._thickness).any():
        print('Inf in thickness')
        break

    fa.run_one_step()

    grid.at_node['topographic__elevation'][grid.core_nodes] += uplift_rate*dt_m
    grid.at_node['aquifer_base__elevation'][grid.core_nodes] += uplift_rate*dt_m - w0*np.exp(-(elev[grid.core_nodes]-base[grid.core_nodes])/dc)*dt_m

    sp.run_one_step(dt_m)
    ld.run_one_step(dt_m)

    elev[elev<base] = base[elev<base]

    if i % output_interval == 0:
        gw_flux[:] = gdp.calc_gw_flux_at_node()

        filename = '../DupuitLEMOutput/grid_'+str(i)+'.nc'
        filenames.append(filename)
        time_list.append(i*dt_m/(365*24*3600))
        write_raster_netcdf(
                filename, grid, names=output_fields, format="NETCDF4")
        print('Completed loop %d' % i)

t1 = time.time()

tot_time = t1-t0

# collect output and save

# open all files as a xarray dataset
ds = xr.open_mfdataset(
    filenames,
    combine='nested',
    concat_dim="nt",
    engine="netcdf4",
    data_vars=output_fields)

# add a time dimension
time_array = np.asarray(time_list)
time_xr = xr.DataArray(
    time_array,
    dims=("nt"),
    attrs={"units": time_unit,
        "standard_name": "time"},
        )

# dimensions and coordinates
ds["time"] = time_xr
ds = ds.set_coords(["x", "y", "time"])
ds = ds.rename(name_dict={"ni": "x", "nj": "y", "nt": "time"})
ds["x"] = xr.DataArray(ds.x, dims=("x"), attrs={"units": space_unit})
ds["y"] = xr.DataArray(ds.y, dims=("y"), attrs={"units": space_unit})

# write output
filename = '../DupuitLEMOutput/grid_collected.nc'
ds.to_netcdf(filename, engine="netcdf4", format="NETCDF4")
ds.close()

try:
    for f in filenames:
        os.remove(f)
except:
    print("Cannot remove files")

#%% Pickle results

# pickle.dump(grid, open('./Dupuit_LEM_results/grid_'+str(time.time())+'.p','wb'))
