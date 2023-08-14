# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:29:33 2022

@author: dgbli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.io.netcdf import from_netcdf
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    )
from DupuitLEM.auxiliary_models import HydrologyEventVadoseStreamPower, SchenkVadoseModel

directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'stoch_ai_sigma_6'
model_runs = np.arange(35)

dfs = []
for ID in model_runs:
    df = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    dfs.append(df)
df_params = pd.concat(dfs, axis=1, ignore_index=True).T
df_params.to_csv('%s/%s/params.csv'%(directory,base_output_path), index=True, float_format='%.3e')

#%%

ID=2

########## Load

# grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, ID))
grid = from_netcdf('%s/%s/stoch_sp_svm_%d_grid_79999.nc'%(directory, base_output_path, ID))
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
# wt = 0.5*(elev-base) + base
wt = grid.at_node['water_table__elevation']


########## Run hydrological model
# load parameters and save just this ID (useful because some runs in a group have been redone with diff parameters)

Ks = df_params['ksat'][ID] #hydraulic conductivity [m/s]
n = df_params['ne'][ID] #drainable porosity [-]
b = df_params['b'][ID] #characteristic depth  [m]
p = df_params['p'][ID] #average precipitation rate [m/s]
tg = df_params['tg'][ID]
dtg = df_params['dtg'][ID]
hg = df_params['hg'][ID]

pet = df_params['pet'][ID]
na = df_params['na'][ID]
tr = df_params['tr'][ID] #mean storm duration [s]
tb = df_params['tb'][ID] #mean interstorm duration [s]
ds = df_params['ds'][ID] #mean storm depth [m]
T_h = 100*(tr+tb) #20*df_params['Th'] #total hydrological time [s]
try:
    extraction_tol = df_params['extraction_tol'][ID]
except:
    extraction_tol = 0.0
    
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
method = 'D8'

gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=n,
                                  hydraulic_conductivity=Ks,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.05,
                                  vn_coefficient = 0.05,
                                  )
pdr = PrecipitationDistribution(mg, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=T_h)
pdr.seed_generator(seedval=2)
svm = SchenkVadoseModel(
                    potential_evapotranspiration_rate=pet,
                    available_water_content=0.2,
                    profile_depth=b,
                    
                    num_bins=10000,
                    )
svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
# if extraction_tol>0:
#     svm.set_max_extraction_depth(ds, tr, tb, threshold=extraction_tol)
# svm.extraction_depth_mask = svm.depths > 0.1
hm = HydrologyEventVadoseStreamPower(
                                    mg,
                                    precip_generator=pdr,
                                    groundwater_model=gdp,
                                    vadose_model=svm,
                                    )

#run model
hm.run_step_record_state()

#%% Budyko figures

df_output = {}
df_output['recharge_efficiency'] = hm.cum_recharge / hm.cum_precip
df_output['(P-Q-Qgw)/P'] = (hm.cum_precip - hm.cum_runoff - hm.cum_gw_export)/hm.cum_precip
df_output['(P-Q)/P'] = (hm.cum_precip - hm.cum_runoff)/hm.cum_precip
df_output['(P-R)/P'] = (hm.cum_precip - hm.cum_recharge)/hm.cum_precip
df_output['Q/P'] = hm.cum_runoff/hm.cum_precip

plt.figure()
plt.scatter(df_params['ai'][ID], df_output['(P-Q-Qgw)/P'], label='(P-Q-Qgw)/P')
plt.scatter(df_params['ai'][ID], df_output['(P-Q)/P'], label='(P-Q)/P' )
plt.scatter(df_params['ai'][ID], df_output['(P-R)/P'], label='(P-R)/P' )
plt.plot([0,1], [0,1], 'k--')
plt.plot([1,2], [1,1], 'k--')
plt.xlabel('Ai')
plt.ylabel('Index')
# plt.ylabel('(R-Q)/R')
plt.legend()


plt.figure()
plt.scatter(df_params['ai'][ID], df_output['(P-Q-Qgw)/P'])
plt.scatter(df_params['ai'][ID], df_output['(P-Q)/P'])
# plt.scatter(df_params['ai'], alt_metric)
plt.plot([0,1], [0,1], 'k--')   
plt.plot([1,2], [1,1], 'k--')
plt.xlabel('Ai')
plt.ylabel('(P-Q)/P')


plt.figure()
plt.scatter(svm.sat_profile, svm.depths)



#%% Testing whether recharge problem appears in recharge/extraction with depth

ID=30

Ks = df_params['ksat'][ID] #hydraulic conductivity [m/s]
n = df_params['ne'][ID] #drainable porosity [-]
b = df_params['b'][ID] #characteristic depth  [m]
p = df_params['p'][ID] #average precipitation rate [m/s]
tg = df_params['tg'][ID]
dtg = df_params['dtg'][ID]
hg = df_params['hg'][ID]

pet = df_params['pet'][ID]
na = df_params['na'][ID]
tr = df_params['tr'][ID] #mean storm duration [s]
tb = df_params['tb'][ID] #mean interstorm duration [s]
ds = df_params['ds'][ID] #mean storm depth [m]


svm = SchenkVadoseModel(
                    potential_evapotranspiration_rate=pet,
                    available_water_content=0.2,
                    profile_depth=b,
                    num_bins=10000,
                    )
svm.run_model(num_timesteps=10000,
              mean_storm_depth=ds,
              mean_storm_duration=tr,
              mean_interstorm_duration=tb,
              random_seed=1234
              )

mean_pet_rate = svm.cum_extraction/svm.cum_interstorm_dt
mean_recharge_rate = svm.cum_recharge/svm.cum_storm_dt

A = (svm.cum_recharge + svm.cum_extraction)/svm.cum_precip

#%%
plt.figure()
plt.plot(A, svm.depths)
# plt.xlim(0.49,0.53)
plt.ylim((18,0))
plt.ylabel('Depth')
plt.xlabel('(R-E)/P')

Ai = ((pet*tb)/(p*(tr+tb)))
B = (1-A)/Ai


plt.figure()
plt.plot(-mean_pet_rate,svm.depths)
plt.plot(np.ones_like(svm.depths)*pet, svm.depths, label="PET (m/s)")
plt.ylim((18,0))
plt.ylabel('Depth')
plt.xlabel('ET (m/s)')
plt.legend()

plt.figure()
plt.plot(mean_recharge_rate,svm.depths)
plt.plot(np.ones_like(svm.depths)*(ds/tr), svm.depths, label='mean intensity (m/s)')
plt.ylabel('Depth')
plt.xlabel('R (m/s)')
plt.ylim((18,0))
plt.legend()

plt.figure()
plt.plot(svm.recharge_frequency,svm.depths)
# plt.plot(np.ones_like(svm.depths)*(ds/tr), svm.depths, label='mean intensity (m/s)')
plt.ylabel('Depth')
plt.xlabel('R (m/s)')
plt.ylim((18,0))
plt.legend()

#%% one-node coupled model

ID = 30

Ks = df_params['ksat'][ID] #hydraulic conductivity [m/s]
n = df_params['ne'][ID] #drainable porosity [-]
b = df_params['b'][ID] #characteristic depth  [m]
p = df_params['p'][ID] #average precipitation rate [m/s]
tg = df_params['tg'][ID]
dtg = df_params['dtg'][ID]
hg = df_params['hg'][ID]

pet = df_params['pet'][ID]
na = df_params['na'][ID]
tr = df_params['tr'][ID] #mean storm duration [s]
tb = df_params['tb'][ID] #mean interstorm duration [s]
ds = df_params['ds'][ID] #mean storm depth [m]
T_h = 1000*(tr+tb) #20*df_params['Th'] #total hydrological time [s]

#initialize grid
mg = RasterModelGrid((3,3), xy_spacing=grid.dx)
mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_CLOSED, top=mg.BC_NODE_IS_CLOSED, \
                              left=mg.BC_NODE_IS_FIXED_VALUE, bottom=mg.BC_NODE_IS_CLOSED)
z = mg.add_zeros('node', 'topographic__elevation')
z[:] = b
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = 0
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = z.copy()

#initialize components
method = 'D8'

gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=n,
                                  hydraulic_conductivity=Ks,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.05,
                                  vn_coefficient = 0.05,
                                  )
pdr = PrecipitationDistribution(mg, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=T_h)
pdr.seed_generator(seedval=2)
svm = SchenkVadoseModel(
                    potential_evapotranspiration_rate=pet,
                    available_water_content=0.2,
                    profile_depth=b,
                    num_bins=10000,
                    )
hm = HydrologyEventVadoseStreamPower(
                                    mg,
                                    precip_generator=pdr,
                                    groundwater_model=gdp,
                                    vadose_model=svm,
                                    )

#run model
hm.run_step_record_state()

#%%

df_output = {}
df_output['recharge_efficiency'] = hm.cum_recharge / hm.cum_precip
df_output['(P-Q-Qgw)/P'] = (hm.cum_precip - hm.cum_runoff - hm.cum_gw_export)/hm.cum_precip
df_output['(P-Q)/P'] = (hm.cum_precip - hm.cum_runoff)/hm.cum_precip
df_output['(P-R)/P'] = (hm.cum_precip - hm.cum_recharge)/hm.cum_precip
df_output['Q/P'] = hm.cum_runoff/hm.cum_precip

plt.figure()
plt.scatter(df_params['ai'][ID], df_output['(P-Q-Qgw)/P'], label='(P-Q-Qgw)/P')
plt.scatter(df_params['ai'][ID], df_output['(P-Q)/P'], label='(P-Q)/P' )
plt.scatter(df_params['ai'][ID], df_output['(P-R)/P'], label='(P-R)/P' )
plt.plot([0,1], [0,1], 'k--')
plt.plot([1,2], [1,1], 'k--')
plt.xlabel('Ai')
plt.ylabel('Index')
# plt.ylabel('(R-Q)/R')
plt.legend()