"""
Analysis of results on HPC for steady stream power model runs.
"""

import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from landlab import imshow_grid
from landlab.io.netcdf import read_netcdf, from_netcdf, to_netcdf

from landlab import RasterModelGrid, LinkStatus
from landlab.components import (
    GroundwaterDupuitPercolator,
    HeightAboveDrainageCalculator,
    DrainageDensity,
    )
from landlab.grid.mappers import map_downwind_node_link_max_to_node
from DupuitLEM.auxiliary_models import HydrologySteadyStreamPower

def calc_max_gw_flux(grid, k, b):
    """
    Calculate the maximum groundwater flux into and out of nodes.
    """
    base = grid.at_node['aquifer_base__elevation']

    # Calculate gradients
    base_grad = grid.calc_grad_at_link(base)
    cosa = np.cos(np.arctan(base_grad))
    hydr_grad = base_grad * cosa

    # calc max gw flux at links
    # Calculate groundwater velocity
    vel = -k * hydr_grad
    vel[grid.status_at_link == LinkStatus.INACTIVE] = 0.0

    # Calculate specific discharge
    q = grid.add_zeros('link', 'q_max_link')
    q[:] = b * cosa * vel

    q_all_links_at_node = q[grid.links_at_node]*grid.link_dirs_at_node
    q_all_links_at_node_dir_out = q_all_links_at_node < 0
    widths = grid.dx*np.ones(q_all_links_at_node.shape)
    Qgw_out = np.sum(q_all_links_at_node*q_all_links_at_node_dir_out*widths, axis=1)

    q_all_links_at_node_dir_in = q_all_links_at_node > 0
    Qgw_in = np.sum(q_all_links_at_node*q_all_links_at_node_dir_in*widths, axis=1)

    return Qgw_in, Qgw_out

def calc_gw_flux(grid):
    """
    Calculate the groundwater flux into and out of nodes.
    """
    # Calculate specific discharge
    q = grid.at_link['groundwater__specific_discharge']

    q_all_links_at_node = q[grid.links_at_node]*grid.link_dirs_at_node
    q_all_links_at_node_dir_out = q_all_links_at_node < 0
    widths = grid.dx*np.ones(q_all_links_at_node.shape)
    Qgw_out = np.sum(q_all_links_at_node*q_all_links_at_node_dir_out*widths, axis=1)

    q_all_links_at_node_dir_in = q_all_links_at_node > 0
    Qgw_in = np.sum(q_all_links_at_node*q_all_links_at_node_dir_in*widths, axis=1)

    return Qgw_in, Qgw_out

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']

# copy original run file
os.system('cp %s.py ../post_proc/%s/%s-%d.py'%(base_output_path[:-2], base_output_path, base_output_path, ID))

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
df_params = pickle.load(open('./parameters.p','rb'))
pickle.dump(df_params, open('../post_proc/%s/parameters.p'%base_output_path,'wb'))

Ks = df_params['ksat'][ID] # hydraulic conductivity [m/s]
p = df_params['p'][ID] # recharge rate [m/s]
n = df_params['n'][ID] # drainable porosity [-]
b = df_params['b'][ID] # characteristic depth  [m]
Th = df_params['Th'][ID] # hydrological timestep

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

gdp = GroundwaterDupuitPercolator(mg,
          porosity=n,
          hydraulic_conductivity=Ks,
          regularization_f=0.01,
          recharge_rate=p,
          courant_coefficient=0.01, #*Ks/1e-5,
          vn_coefficient = 0.01, #*Ks/1e-5,
)

hm = HydrologySteadyStreamPower(
        mg,
        groundwater_model=gdp,
        hydrological_timestep=Th,
        # routing_method='Steepest',
)

#run model
hm.run_step()

##########  Analysis

#dataframe for output
df_output = {}

# Qstar
Q = mg.at_node['surface_water__discharge']
Qstar = mg.add_zeros('node', 'qstar')
Qstar[:] = Q/(mg.at_node['drainage_area']*df_params['p'][ID])

# groundwater flux
q_out_max = mg.add_zeros('node', 'gw_flux_out_max')
q_in_max = mg.add_zeros('node', 'gw_flux_in_max')
q_out = mg.add_zeros('node', 'gw_flux_out')
q_in = mg.add_zeros('node', 'gw_flux_in')
q_in[:], q_out[:] = calc_gw_flux(mg)
q_in_max[:], q_out_max[:] = calc_max_gw_flux(mg, Ks, b)

##### steepness, curvature, and topographic index
S8 = mg.add_zeros('node', 'slope_D8')
S4 = mg.add_zeros('node', 'slope_D4')
curvature = mg.add_zeros('node', 'curvature')
steepness = mg.add_zeros('node', 'steepness')
TI8 = mg.add_zeros('node', 'topographic__index_D8')
TI4 = mg.add_zeros('node', 'topographic__index_D4')

#slope for steepness is the absolute value of D8 gradient associated with
#flow direction. Same as FastscapeEroder. curvature is divergence of gradient.
#Same as LinearDiffuser. TI is done both ways.
dzdx_D8 = mg.calc_grad_at_d8(elev)
dzdx_D4 = mg.calc_grad_at_link(elev)
dzdx_D4[mg.status_at_link == LinkStatus.INACTIVE] = 0.0
S8[:] = abs(dzdx_D8[mg.at_node['flow__link_to_receiver_node']])
S4[:] = map_downwind_node_link_max_to_node(mg, dzdx_D4)

curvature[:] = mg.calc_flux_div_at_node(dzdx_D4)
steepness[:] = np.sqrt(mg.at_node['drainage_area'])*S8
TI8[:] = mg.at_node['drainage_area']/(S8*mg.dx)
TI4[:] = mg.at_node['drainage_area']/(S4*mg.dx)

network = mg.add_zeros('node', 'channel_mask')
network[:] = curvature > 0

######## Calculate HAND
hand = mg.add_zeros('node', 'hand')
hd = HeightAboveDrainageCalculator(mg, channel_mask=network)

hd.run_one_step()
hand[:] = mg.at_node["height_above_drainage__elevation"].copy()
df_output['mean_hand'] = np.mean(hand[mg.core_nodes])
df_output['hand_mean_ridges'] = np.mean(hand[mg.at_node["drainage_area"]==mg.dx**2])

######## Calculate drainage density
dd = DrainageDensity(mg, channel__mask=np.uint8(network))
channel_mask = mg.at_node['channel__mask']
df_output['drainage_density'] = dd.calculate_drainage_density()

####### calculate relief change
output_interval = int(files[1].split('_')[-1][:-3]) - int(files[0].split('_')[-1][:-3])
dt_nd = output_interval*df_params['dtg'][ID]/df_params['tg'][ID]
relief_change = np.zeros(len(files))
for i in range(1,len(files)):
    try:
        grid = from_netcdf(files[i])
    except KeyError:
        grid = read_netcdf(files[i])
    elev = grid.at_node['topographic__elevation']
    relief_change[i] = np.mean(elev[grid.core_nodes])

r_change = pd.DataFrame()
r_change['r_nd'] = relief_change[:]/df_params['hg'][ID]
r_change['drdt_nd'] = np.diff(r_change['r_nd'], prepend=0.0)/dt_nd
r_change['t_nd'] = np.arange(len(files))*dt_nd

####### save things

output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        'at_node:topographic__index_D8',
        'at_node:topographic__index_D4',
        'at_node:channel_mask',
        'at_node:hand',
        'at_node:slope_D8',
        'at_node:slope_D4',
        'at_node:drainage_area',
        'at_node:curvature',
        'at_node:steepness',
        'at_node:qstar',
        'at_node:gw_flux_out',
        'at_node:gw_flux_in',
        'at_node:gw_flux_out_max',
        'at_node:gw_flux_in_max',
        'at_link:groundwater__specific_discharge',
        ]

filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
to_netcdf(mg, filename, include=output_fields, format="NETCDF4")

pickle.dump(df_output, open('../post_proc/%s/output_ID_%d.p'%(base_output_path, ID), 'wb'))
r_change.to_csv('../post_proc/%s/relief_change_%d.csv'%(base_output_path, ID))
