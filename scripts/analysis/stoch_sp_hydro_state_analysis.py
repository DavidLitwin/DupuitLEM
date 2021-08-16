"""
Analysis of results on HPC for stochastic stream power model runs.

update for new stochastic models
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
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
from DupuitLEM.auxiliary_models import HydrologyEventStreamPower

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

########## Run hydrological model
# load parameters and save just this ID (useful because some runs in a group have been redone with diff parameters)
df_params = pickle.load(open('./parameters.p','rb'))
params = df_params.iloc[ID]
pickle.dump(params, open('../post_proc/%s/params_ID_%d.p'%(base_output_path,ID),'wb'))

Ks = df_params['ksat'][ID] #hydraulic conductivity [m/s]
n = df_params['n'][ID] #drainable porosity [-]
b = df_params['b'][ID] #characteristic depth  [m]
p = df_params['p'][ID] #average precipitation rate [m/s]
tr = df_params['tr'][ID] #mean storm duration [s]
tb = df_params['tb'][ID] #mean interstorm duration [s]
ds = df_params['ds'][ID] #mean storm depth [m]
T_h = 10*df_params['Th'][ID] #total hydrological time [s]

#initialize grid
dx = grid.dx
mg = RasterModelGrid(grid.shape, xy_spacing=dx)
mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_CLOSED, top=mg.BC_NODE_IS_CLOSED, \
                              left=mg.BC_NODE_IS_FIXED_VALUE, bottom=mg.BC_NODE_IS_CLOSED)
z = mg.add_zeros('node', 'topographic__elevation')
z[:] = elev
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = base
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = wt

f = open('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), 'w')
def write_SQ(grid, r, dt, file=f):
    cores = grid.core_nodes
    h = grid.at_node["aquifer__thickness"]
    area = grid.cell_area_at_node
    storage = np.sum(n*h[cores]*area[cores])

    qs = grid.at_node["surface_water__specific_discharge"]
    qs_tot = np.sum(qs[cores]*area[cores])
    qs_nodes = np.sum(qs[cores]>1e-10)

    r_tot = np.sum(r[cores]*area[cores])

    file.write('%f, %f, %f, %f, %f\n'%(dt, r_tot, qs_tot, storage, qs_nodes))

#initialize components
if isinstance(mg, RasterModelGrid):
    method = 'D8'
elif isinstance(mg, HexModelGrid):
    method = 'Steepest'
else:
    raise TypeError("grid should be Raster or Hex")

gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=n,
                                  hydraulic_conductivity=Ks,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.05,
                                  vn_coefficient = 0.05,
                                  callback_fun = write_SQ,
                                  )
pdr = PrecipitationDistribution(mg, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=T_h)
pdr.seed_generator(seedval=2)

hm = HydrologyEventStreamPower(
        mg,
        routing_method=method,
        precip_generator=pdr,
        groundwater_model=gdp,
)

#run model
hm.run_step_record_state()
f.close()

##########  Analysis

#dataframe for output
df_output = {}

#load the full storage discharge dataset that was just generated
df = pd.read_csv('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), sep=',',header=None, names=['dt','i', 'qs', 'S', 'qs_cells'])
# remove the first row (test row written by init of gdp)
df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)
# make dimensionless timeseries
Atot = np.sum(mg.cell_area_at_node[mg.core_nodes])
df['qs_star'] = df['qs']/(p*Atot)
df['S_star'] = df['S']/(b*n*Atot)

###### spatial runoff related quantities

# effective Qstar
Q_all = hm.Q_all[1:,:]
dt = np.diff(hm.time)
intensity = hm.intensity[:-1]
qstar_mean = mg.add_zeros('node', 'qstar_mean_no_interevent')

# mean Q based on the geomorphic definition - only Q during storm events does geomorphic work
Q_event_sum = np.zeros(Q_all.shape[1])
for i in range(1,len(Q_all)):
    if intensity[i] > 0.0:
        Q_event_sum += 0.5*(Q_all[i,:]+Q_all[i-1,:])*dt[i]
qstar_mean[:] = (Q_event_sum/np.sum(dt[1:]))/(mg.at_node['drainage_area']*df_params['p'][ID])
qstar_mean[np.isnan(qstar_mean)] = 0.0

# mean and variance of water table
wt_all = hm.wt_all[1:,:]
base_all = np.ones(wt_all.shape)*mg.at_node['aquifer_base__elevation']
elev_all = np.ones(wt_all.shape)*mg.at_node['topographic__elevation']
wtrel_all = np.zeros(wt_all.shape)
wtrel_all[:, mg.core_nodes] = (wt_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])/(elev_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])

# water table and saturation at end of storm and interstorm
thresh = 1e-10 #np.mean(mg.cell_area_at_node[grid.core_nodes])*df_params['p'][ID]
sat_all = (Q_all > thresh)
wtrel_end_interstorm = mg.add_zeros('node', 'wtrel_mean_end_interstorm')
wtrel_end_storm = mg.add_zeros('node', 'wtrel_mean_end_storm')
sat_end_interstorm = mg.add_zeros('node', 'sat_mean_end_interstorm')
sat_end_storm = mg.add_zeros('node', 'sat_mean_end_storm')
Q_end_interstorm = mg.add_zeros('node', 'Q_mean_end_interstorm')
Q_end_storm = mg.add_zeros('node', 'Q_mean_end_storm')

wtrel_end_storm[:] = np.mean(wtrel_all[intensity>0,:], axis=0)
wtrel_end_interstorm[:] = np.mean(wtrel_all[intensity==0.0,:], axis=0)
sat_end_storm[:] = np.mean(sat_all[intensity>0,:], axis=0)
sat_end_interstorm[:] = np.mean(sat_all[intensity==0.0,:], axis=0)
Q_end_storm[:] = np.mean(Q_all[intensity>0,:], axis=0)
Q_end_interstorm[:] = np.mean(Q_all[intensity==0.0,:], axis=0)



####### save things

output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        'at_node:channel_mask_storm',
        'at_node:channel_mask_interstorm',
        'at_node:hand_storm',
        'at_node:hand_interstorm',
        'at_node:topographic__index_D8',
        'at_node:topographic__index_D4',
        'at_node:slope_D8',
        'at_node:slope_D4',
        'at_node:drainage_area',
        'at_node:curvature',
        'at_node:steepness',
        'at_node:qstar_mean_no_interevent',
        'at_node:wtrel_mean_end_storm',
        'at_node:wtrel_mean_end_interstorm',
        'at_node:sat_mean_end_storm',
        'at_node:sat_mean_end_interstorm',
        'at_node:Q_mean_end_storm',
        'at_node:Q_mean_end_interstorm',

        ]

filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
to_netcdf(mg, filename, include=output_fields, format="NETCDF4")

pickle.dump(df_output, open('../post_proc/%s/output_ID_%d.p'%(base_output_path, ID), 'wb'))
pickle.dump(df, open('../post_proc/%s/q_s_dt_ID_%d.p'%(base_output_path, ID), 'wb'))
