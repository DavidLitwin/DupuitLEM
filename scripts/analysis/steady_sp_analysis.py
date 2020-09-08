"""
Analysis of results on HPC for steady stream power model runs.
"""

import os
import glob
from re import sub
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from landlab import imshow_grid
from landlab.io.netcdf import to_netcdf, from_netcdf

from landlab import RasterModelGrid, HexModelGrid, LinkStatus
from landlab.components import (
    GroundwaterDupuitPercolator,
    FlowDirectorD8,
    FlowDirectorSteepest,
    FlowAccumulator,
    LakeMapperBarnes,
    HeightAboveDrainageCalculator,
    DrainageDensity,
    )
from DupuitLEM.auxiliary_models import HydrologySteadyStreamPower

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']

########## Load and basic plot
grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:float(sub("\D", "", x[25:-3])))

path = files[-1]
iteration = int(sub("\D", "", path[25:-3]))

mg = from_netcdf(path)
elev = mg.at_node['topographic__elevation']
base = mg.at_node['aquifer_base__elevation']
wt = mg.at_node['water_table__elevation']

# elevation
plt.figure(figsize=(8,6))
imshow_grid(mg, elev, cmap='gist_earth', colorbar_label='Elevation [m]', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID,iteration))
plt.savefig('../post_proc/%s/elev_ID_%d.png'%(base_output_path, ID))
plt.close()


########## Run hydrological model
df_params = pickle.load(open('./parameters.p','rb'))
df_params['hg'] = df_params['U']/df_params['K']
df_params['lg'] = np.sqrt(df_params['D']/df_params['K'])
df_params['tg'] = 1/df_params['K']
pickle.dump(df_params, open('../post_proc/%s/parameters.p'%base_output_path,'wb'))

Ks = df_params['ksat'][ID] # hydraulic conductivity [m/s]
p = df_params['p'][ID] # recharge rate [m/s]
n = df_params['n'][ID] # drainable porosity [-]
b = df_params['b'][ID] # characteristic depth  [m]
Th = df_params['Th'][ID] # hydrological timestep

# initialize components
if isinstance(mg, RasterModelGrid):
    method = 'D8'
    fd = FlowDirectorD8(mg)
elif isinstance(mg, HexModelGrid):
    method = 'Steepest'
    fd = FlowDirectorSteepest(mg)
else:
    raise TypeError("grid should be Raster or Hex")

gdp = GroundwaterDupuitPercolator(mg,
          porosity=n,
          hydraulic_conductivity=Ks,
          regularization_f=0.01,
          recharge_rate=p,
          courant_coefficient=0.01*Ks/1e-5,
          vn_coefficient = 0.01*Ks/1e-5,
)
fa = FlowAccumulator(mg,
				        surface='topographic__elevation',
						flow_director=fd,
						runoff_rate='average_surface_water__specific_discharge')
lmb = LakeMapperBarnes(mg, method=method, fill_flat=False,
						  surface='topographic__elevation',
						  fill_surface='topographic__elevation',
						  redirect_flow_steepest_descent=False,
						  reaccumulate_flow=False,
						  track_lakes=False,
						  ignore_overfill=True)
hm = HydrologySteadyStreamPower(
        mg,
        groundwater_model=gdp,
        flow_director=fd,
        flow_accumulator=fa,
        lake_mapper=lmb,
        hydrological_timestep=Th,
)

#run model
hm.run_step()
# N = int(5e3) # max number of timesteps to take
# wt_max_change = np.zeros(N)
# for i in range(N):
#     zwt_0 = zwt.copy()
#     hm.run_step()
#     wt_max_change = max(abs(zwt_0-zwt)/Th)

    # if wt_max_change < 1e-14:
    #     break

##########  Analysis

#dataframe for output
df_output = {}

##### channel network

# Qstar
Q = mg.at_node['surface_water__discharge']
Qstar = mg.add_zeros('node', 'qstar')
Qstar[:] = Q/(mg.at_node['drainage_area']*df_params['p'][ID])

#find number of saturated cells
Q_nodes = Q > p*mg.cell_area_at_node

#set fields
network = mg.add_zeros('node', 'channel_mask')
network[:] = Q_nodes.copy()

##### steepness and curvature
S = mg.add_zeros('node', 'slope')
A = mg.at_node['drainage_area']
curvature = mg.add_zeros('node', 'curvature')
steepness = mg.add_zeros('node', 'steepness')

if isinstance(mg, RasterModelGrid):
    #slope is the absolute value of D8 gradient associated with flow direction. Same as FastscapeEroder.
    #curvature is divergence of gradient. Same as LinearDiffuser.
    dzdx_D8 = mg.calc_grad_at_d8(elev)
    dzdx_D4 = mg.calc_grad_at_link(elev)
    dzdx_D4[mg.status_at_link == LinkStatus.INACTIVE] = 0.0
    S[:] = abs(dzdx_D8[mg.at_node['flow__link_to_receiver_node']])

    curvature[:] = mg.calc_flux_div_at_node(dzdx_D4)
    steepness[:] = np.sqrt(A)*S

    width = mg.dx
elif isinstance(mg, HexModelGrid):
    #slope is max of the absolute value of gradient.
    #curvature is divergence of gradient. Same as LinearDiffuser.
    dzdx_D6 = mg.calc_grad_at_link(elev)
    dzdx_D6[mg.status_at_link == LinkStatus.INACTIVE] = 0.0
    S[:] = abs(dzdx_D6[mg.at_node['flow__link_to_receiver_node']])

    curvature[:] = mg.calc_flux_div_at_node(dzdx_D6)
    steepness[:] = np.sqrt(A)*S

    width = mg.spacing
else:
    raise TypeError("grid should be Raster or Hex")

######## Calculate HAND
hand = mg.add_zeros('node', 'hand')
hd = HeightAboveDrainageCalculator(mg, channel_mask=network)

hd.run_one_step()
hand[:] = mg.at_node["height_above_drainage__elevation"].copy()
df_output['mean_hand'] = np.mean(hand[mg.core_nodes])

######## Calculate drainage density
if isinstance(mg, RasterModelGrid):
    dd = DrainageDensity(mg, channel__mask=np.uint8(network))
    channel_mask = mg.at_node['channel__mask']
    df_output['drainage_density'] = dd.calculate_drainage_density()

######## Topogrpahic index
TI = mg.add_zeros('node', 'topographic__index')
S = mg.calc_slope_at_node(elev)
TI[:] = mg.at_node['drainage_area']/(S*width)

####### calculate elevation change
z_change = np.zeros((len(files),5))
grid = from_netcdf(files[0])
elev0 = grid.at_node['topographic__elevation']
for i in range(1,len(files)):

    grid = from_netcdf(files[i])
    elev = grid.at_node['topographic__elevation']

    elev_diff = abs(elev-elev0)
    z_change[i,0] = np.max(elev_diff)
    z_change[i,1] = np.percentile(elev_diff,90)
    z_change[i,2] = np.percentile(elev_diff,50)
    z_change[i,3] = np.percentile(elev_diff,10)
    z_change[i,4] = np.mean(elev_diff)

    elev0 = elev.copy()

df_z_change = pd.DataFrame(z_change,columns=['max', '90 perc', '50 perc', '10 perc', 'mean'])

####### save things

output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        'at_node:topographic__index',
        'at_node:channel_mask',
        'at_node:hand',
        'at_node:slope',
        'at_node:drainage_area',
        'at_node:curvature',
        'at_node:steepness',
        'at_node:qstar',
        ]

filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
to_netcdf(mg, filename, include=output_fields, format="NETCDF4")

pickle.dump(df_output, open('../post_proc/%s/output_ID_%d.p'%(base_output_path, ID), 'wb'))
pickle.dump(df_z_change, open('../post_proc/%s/z_change_%d.p'%(base_output_path, ID), 'wb'))
