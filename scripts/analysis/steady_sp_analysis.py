"""
Analysis of results on HPC for steady stream power model runs.
"""

import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from landlab import imshow_grid
from landlab.io.netcdf import from_netcdf, to_netcdf

from landlab import RasterModelGrid, HexModelGrid, LinkStatus
from landlab.components import (
    GroundwaterDupuitPercolator,
    HeightAboveDrainageCalculator,
    DrainageDensity,
    ChiFinder,
    )
from landlab.grid.mappers import map_downwind_node_link_max_to_node
from DupuitLEM.auxiliary_models import HydrologySteadyStreamPower
from DupuitLEM.grid_functions import calc_gw_flux, calc_max_gw_flux

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']

# copy original run file
os.system('cp *.py ../post_proc/%s/%s-%d.py'%(base_output_path, base_output_path, ID))

########## Load and basic plot
grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
iteration = int(files[-1].split('_')[-1][:-3])

mg = from_netcdf(files[-1])
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
try:
    df_params = pd.read_csv('parameters.csv', index_col=0)[task_id]
    df_params.to_csv('../post_proc/%s/params_ID_%d.csv'%(base_output_path,ID), index=True)
except FileNotFoundError:
    df_params = pickle.load(open('./parameters.p','rb'))
    df_params = df_params.iloc[ID]
    df_params.to_csv('../post_proc/%s/params_ID_%d.csv'%(base_output_path,ID), index=True)

Ks = df_params['ksat'] # hydraulic conductivity [m/s]
p = df_params['p'] # recharge rate [m/s]
ne = df_params['ne'] # drainable porosity [-]
b = df_params['b'] # characteristic depth  [m]
Th = df_params['Th'] # hydrological timestep
dtg = df_params['dtg']
tg = df_params['tg']
hg = df_params['hg']
conc = 0.5 # concavity for chi analysis (default...)
try:
    RE = df_params['RE']
except KeyError:
    print("no recharge efficiency 'RE' provided. Using RE = 1.0")
    RE = 1.0

gdp = GroundwaterDupuitPercolator(mg,
        porosity=ne,
        hydraulic_conductivity=Ks,
        regularization_f=0.01,
        recharge_rate=p*RE,
        courant_coefficient=0.1, #*Ks/1e-5,
        vn_coefficient=0.1, #*Ks/1e-5,
)

hm = HydrologySteadyStreamPower(
        mg,
        groundwater_model=gdp,
        hydrological_timestep=Th,
        routing_method='D8' if isinstance(mg, RasterModelGrid) else 'Steepest',
)

#run model
for i in tqdm(range(500), desc="Completion"):
    hm.run_step()

##########  Analysis

#dataframe for output
df_output = {}

# Qstar
Q = mg.at_node['surface_water__discharge']
Qstar = mg.add_zeros('node', 'qstar')
Qstar[:] = Q/(mg.at_node['drainage_area']*p)

# groundwater flux
q_out_max = mg.add_zeros('node', 'gw_flux_out_max')
q_in_max = mg.add_zeros('node', 'gw_flux_in_max')
q_out = mg.add_zeros('node', 'gw_flux_out')
q_in = mg.add_zeros('node', 'gw_flux_in')
q_in[:], q_out[:] = calc_gw_flux(mg)
q_in_max[:], q_out_max[:] = calc_max_gw_flux(mg, Ks, b)

##### steepness, curvature, and topographic index
curvature = mg.add_zeros('node', 'curvature')
steepness = mg.add_zeros('node', 'steepness')

if isinstance(mg, RasterModelGrid):
    S8 = mg.add_zeros('node', 'slope_D8')
    S4 = mg.add_zeros('node', 'slope_D4')
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
    dx = max(mg.length_of_face)
    TI8[:] = mg.at_node['drainage_area']/(S8*dx)
    TI4[:] = mg.at_node['drainage_area']/(S4*dx)
else:
    S6 = mg.add_zeros('node', 'slope_D6')
    TI6 = mg.add_zeros('node', 'topographic__index_D6')

    dzdx_D6 = mg.calc_grad_at_link(elev)
    dzdx_D6[mg.status_at_link == LinkStatus.INACTIVE] = 0.0
    S6[:] = abs(dzdx_D6[mg.at_node['flow__link_to_receiver_node']])

    curvature[:] = mg.calc_flux_div_at_node(dzdx_D6)
    steepness[:] = np.sqrt(mg.at_node['drainage_area'])*S6
    dx = max(mg.length_of_face)
    TI6[:] = mg.at_node['drainage_area']/(S6*dx)

network = mg.add_zeros('node', 'channel_mask')
network[:] = curvature > 0

######## Calculate HAND
hand = mg.add_zeros('node', 'hand')
hd = HeightAboveDrainageCalculator(mg, channel_mask=network)

hd.run_one_step()
hand[:] = mg.at_node["height_above_drainage__elevation"].copy()
df_output['mean hand'] = np.mean(hand[mg.core_nodes])
cell_area = max(mg.cell_area_at_node)
df_output['hand mean ridges'] = np.mean(hand[mg.at_node["drainage_area"]==cell_area])

######## Calculate drainage density, Chi
if isinstance(mg, RasterModelGrid):
    dd = DrainageDensity(mg, channel__mask=np.uint8(network))
    channel_mask = mg.at_node['channel__mask']
    df_output['drainage density'] = dd.calculate_drainage_density()
    df_output['mean hillslope len'] = 1/(2*df_output['drainage density'])
    df_output['mean hillslope len ridges'] = np.mean(mg.at_node["surface_to_channel__minimum_distance"][mg.at_node["drainage_area"]==cell_area])

   # chi
    cf = ChiFinder(mg, min_drainage_area=mg.dx**2, reference_concavity=conc, reference_area=1)
    cf.calculate_chi()


####### calculate relief change
output_interval = int(files[1].split('_')[-1][:-3]) - int(files[0].split('_')[-1][:-3])
dt_nd = output_interval*dtg/tg
relief_change = np.zeros(len(files))
for i in range(1,len(files)):
    grid = from_netcdf(files[i])
    elev = grid.at_node['topographic__elevation']
    relief_change[i] = np.mean(elev[grid.core_nodes])

r_change = pd.DataFrame()
r_change['r_nd'] = relief_change[:]/hg
r_change['drdt_nd'] = np.diff(r_change['r_nd'], prepend=0.0)/dt_nd
r_change['t_nd'] = np.arange(len(files))*dt_nd

####### save things
raster_out = [
        'at_node:topographic__index_D8',
        'at_node:topographic__index_D4',
        'at_node:slope_D8',
        'at_node:slope_D4',
        'at_node:channel__chi_index',
        ]
hex_out = [
        'at_node:topographic__index_D6',
        'at_node:slope_D6',
        ]

shared_out = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        'at_node:channel_mask',
        'at_node:hand',
        'at_node:drainage_area',
        'at_node:curvature',
        'at_node:steepness',
        'at_node:qstar',
        'at_node:gw_flux_out',
        'at_node:gw_flux_in',
        'at_node:gw_flux_out_max',
        'at_node:gw_flux_in_max',
        'at_link:groundwater__specific_discharge',
        'at_node:surface_water__discharge',
        ]

output_fields = shared_out+raster_out if isinstance(mg, RasterModelGrid) else shared_out + hex_out
filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
to_netcdf(mg, filename, include=output_fields, format="NETCDF4")

df_output = pd.DataFrame.from_dict(df_output, orient='index', columns=[ID])
df_output.to_csv('../post_proc/%s/output_ID_%d.csv'%(base_output_path, ID))
r_change.to_csv('../post_proc/%s/relief_change_%d.csv'%(base_output_path, ID))
