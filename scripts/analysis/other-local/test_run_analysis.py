# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:39:14 2022

@author: dgbli
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from landlab import imshow_grid, RasterModelGrid, LinkStatus
from landlab.io.netcdf import to_netcdf, from_netcdf, read_netcdf
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    )
from landlab.grid.mappers import map_downwind_node_link_max_to_node
from DupuitLEM.auxiliary_models import HydrologyEventVadoseStreamPower, SchenkVadoseModel


directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'stoch_gam_sigma_14' #'stoch_gam_sigma_9' #
filename = 'stoch_sp_svm_24_grid_7999.nc'
ID = 24

df_params = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)[str(ID)]

#%%
grid = from_netcdf(os.path.join(directory,base_output_path, filename))
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt = grid.at_node['water_table__elevation']

Ks = df_params['ksat'] #hydraulic conductivity [m/s]
ne = df_params['ne'] #drainable porosity [-]
b = df_params['b'] #characteristic depth  [m]
p = df_params['p'] #average precipitation rate [m/s]
tg = df_params['tg']
dtg = df_params['dtg']
hg = df_params['hg']

pet = df_params['pet']
na = df_params['na'] #plant available volumetric water content
tr = df_params['tr'] #mean storm duration [s]
tb = df_params['tb'] #mean interstorm duration [s]
ds = df_params['ds'] #mean storm depth [m]
T_h = 2000*(tr+tb) #20*df_params['Th'] #total hydrological time [s]

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

method = 'D8'

        
#%%
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
pdr.seed_generator(seedval=4)
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
# hm.run_step()

# f = open('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), 'w')
# def write_SQ(grid, r, dt, file=f):
#     cores = grid.core_nodes
#     h = grid.at_node["aquifer__thickness"]
#     wt = grid.at_node["water_table__elevation"]
#     z = grid.at_node["topographic__elevation"]
#     sat = (z-wt) < sat_cond*hg
#     qs = grid.at_node["surface_water__specific_discharge"]
#     area = grid.cell_area_at_node

#     storage = np.sum(ne*h[cores]*area[cores])
#     qs_tot = np.sum(qs[cores]*area[cores])
#     sat_nodes = np.sum(sat[cores])
#     r_tot = np.sum(r[cores]*area[cores])

#     file.write('%f, %f, %f, %f, %f\n'%(dt, r_tot, qs_tot, storage, sat_nodes))
# gdp.callback_fun = write_SQ

# hm.run_step_record_state()
# f.close()


#%%

hm.generate_exp_precip()

areas = grid.cell_area_at_node[grid.core_nodes]

cum_precip = 0
cum_pet = 0

for i in range(len(hm.storm_dts)):
    cum_precip += np.sum(hm.intensities[i] * areas) * hm.storm_dts[i]
    cum_pet += np.sum(hm.svm.pet * areas) * hm.interstorm_dts[i]


aip = cum_pet/cum_precip

#%%
sat_cond = 0.025 

wt_all = hm.wt_all[1:,:]
base_all = np.ones(wt_all.shape)*mg.at_node['aquifer_base__elevation']
elev_all = np.ones(wt_all.shape)*mg.at_node['topographic__elevation']
wtrel_all = np.zeros(wt_all.shape)
wtrel_all[:, mg.core_nodes] = (wt_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])/(elev_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])

# water table and saturation at end of storm and interstorm
sat_all = (elev_all-wt_all) < sat_cond*hg

sat_timeseries = np.mean(sat_all, axis=1)
sort = np.argsort(sat_timeseries)

quantiles = np.quantile(sat_timeseries, np.array([0, 0.25, 0.50, 0.75, 0.95, 0.99]))
inds = [(np.abs(sat_timeseries - i)).argmin() for i in quantiles]

results = sat_all[inds, :]
dx = grid.dx/df_params['lg']
y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
fig, axs= plt.subplots(ncols=len(quantiles))
for i in range(len(quantiles)):

    im = axs[i].imshow(results[i,:].reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap="viridis_r",
                         interpolation = 'none'
                         )

    
#%%

recharge_event = mg.add_zeros('node', 'recharge_rate_mean_storm')
recharge_event[:] = np.mean(hm.r_all[range(0,hm.r_all.shape[0],2),:], axis=0)

df_output = {}
df_output['cum_precip'] = hm.cum_precip
df_output['cum_recharge'] = hm.cum_recharge
df_output['cum_runoff'] = hm.cum_runoff
df_output['cum_extraction'] = hm.cum_extraction
df_output['cum_gw_export'] = hm.cum_gw_export
df_output['cum_pet'] = hm.cum_pet

"""ratio of total recharge to total precipitation, averaged over space and time.
this accounts for time varying recharge with precipitation rate, unsat
storage and ET, as well as spatially variable recharge with water table depth.
"""
df_output['recharge_efficiency'] = hm.cum_recharge / hm.cum_precip
df_output['(P-Q-Qgw)/P'] = (hm.cum_precip - hm.cum_runoff - hm.cum_gw_export)/hm.cum_precip
df_output['Q/P'] = hm.cum_runoff/hm.cum_precip

df_output['cum_et_calc'] = df_output['cum_precip'] - df_output['cum_runoff'] - df_output['cum_gw_export']
df_output['ETgw/ET'] =  1 - (df_output['cum_et_calc'] - (-df_output['cum_extraction']))/df_output['cum_et_calc']
