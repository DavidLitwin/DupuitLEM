"""
Analysis of results on HPC for stochastic stream power model runs, only
up to the callback.

"""

import os
import csv
import glob
import pickle
import numpy as np
import pandas as pd
import richdem as rd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from landlab import imshow_grid, RasterModelGrid, HexModelGrid, LinkStatus
from landlab.io.netcdf import to_netcdf, from_netcdf, read_netcdf
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    HeightAboveDrainageCalculator,
    DrainageDensity,
    )
from landlab.grid.mappers import map_downwind_node_link_max_to_node
from DupuitLEM.auxiliary_models import HydrologyEventVadoseStreamPower, SchenkVadoseModel

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']

########## Load and basic plot
grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
iteration = int(files[-1].split('_')[-1][:-3])

try:
    grid = from_netcdf(files[-1])
except KeyError:
    grid = read_netcdf(files[-1])
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt = grid.at_node['water_table__elevation']

# elevation
plt.figure(figsize=(8,6))
imshow_grid(grid, elev, cmap='gist_earth', colorbar_label='Elevation [m]', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID,iteration))
plt.savefig('../post_proc/%s/elev_ID_%d.png'%(base_output_path, ID))
plt.close()


########## Run hydrological model
# load parameters and save just this ID (useful because some runs in a group have been redone with diff parameters)
try:
    df_params = pd.read_csv('parameters.csv', index_col=0)[task_id]
    df_params.to_csv('../post_proc/%s/params_ID_%d.csv'%(base_output_path,ID), index=True)
except FileNotFoundError:
    df_params = pickle.load(open('./parameters.p','rb'))
    df_params = df_params.iloc[ID]
    df_params.to_csv('../post_proc/%s/params_ID_%d.csv'%(base_output_path,ID), index=True)

Ks = df_params['ksat'] #hydraulic conductivity [m/s]
ne = df_params['ne'] #drainable porosity [-]
b = df_params['b'] #characteristic depth  [m]
p = df_params['p'] #average precipitation rate [m/s]
tg = df_params['tg']
dtg = df_params['dtg']
hg = df_params['hg']
try:
    pet = df_params['pet']
    na = df_params['na'] #plant available volumetric water content
    tr = df_params['tr'] #mean storm duration [s]
    tb = df_params['tb'] #mean interstorm duration [s]
    ds = df_params['ds'] #mean storm depth [m]
    T_h = 500*(tr+tb) #20*df_params['Th'] #total hydrological time [s]
except KeyError:
    df_params_1d = pd.read_csv('df_params_1d_%d.csv'%ID, index_col=0)[task_id]
    pet = df_params_1d['pet']
    na = df_params['na']
    tr = df_params_1d['tr'] #mean storm duration [s]
    tb = df_params_1d['tb'] #mean interstorm duration [s]
    ds = df_params_1d['ds'] #mean storm depth [m]
    T_h = 500*(tr+tb) #df_params_1d['Nt']*(tr+tb) #total hydrological time [s]

sat_cond = 0.025 # distance from surface (units of hg) for saturation

#initialize grid
mg = RasterModelGrid(grid.shape, xy_spacing=grid.dx)
mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_CLOSED, top=mg.BC_NODE_IS_CLOSED, \
                              left=mg.BC_NODE_IS_FIXED_VALUE, bottom=mg.BC_NODE_IS_CLOSED)
z = mg.add_zeros('node', 'topographic__elevation')
z[:] = elev
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = base
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = wt

#initialize components
if isinstance(mg, RasterModelGrid):
    method = 'D8'
elif isinstance(mg, HexModelGrid):
    method = 'Steepest'
else:
    raise TypeError("grid should be Raster or Hex")

gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=ne,
                                  hydraulic_conductivity=Ks,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.05,
                                  vn_coefficient = 0.05,
                                  #callback_fun = write_SQ,
                                  )
pdr = PrecipitationDistribution(mg, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=T_h)
pdr.seed_generator(seedval=2)
svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                 available_water_content=na,
                 profile_depth=b,
                 num_bins=500,
                 )
svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
hm = HydrologyEventVadoseStreamPower(
                                    mg,
                                    precip_generator=pdr,
                                    groundwater_model=gdp,
                                    vadose_model=svm,
                                    )

#run model
hm.run_step()

f = open('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), 'w')
def write_SQ(grid, r, dt, file=f):
    cores = grid.core_nodes
    h = grid.at_node["aquifer__thickness"]
    wt = grid.at_node["water_table__elevation"]
    z = grid.at_node["topographic__elevation"]
    sat = (z-wt) < sat_cond*hg
    qs = grid.at_node["surface_water__specific_discharge"]
    area = grid.cell_area_at_node

    storage = np.sum(ne*h[cores]*area[cores])
    qs_tot = np.sum(qs[cores]*area[cores])
    sat_nodes = np.sum(sat[cores])
    r_tot = np.sum(r[cores]*area[cores])

    file.write('%f, %f, %f, %f, %f\n'%(dt, r_tot, qs_tot, storage, sat_nodes))

f1 = open('../post_proc/%s/sat_%d'%(base_output_path, ID), 'w', newline ='')
w1 = csv.writer(f1)
def write_wt(grid, r, dt, file=w1):
    wt = grid.at_node["water_table__elevation"]
    z = grid.at_node["topographic__elevation"]
    sat = (z-wt) < sat_cond*hg
    file.writerow(sat*1)

gdp.callback_fun = write_wt

hm.run_step_record_state()
f.close()
f1.close()
