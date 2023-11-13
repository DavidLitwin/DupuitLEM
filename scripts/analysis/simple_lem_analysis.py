"""
Analysis of results on HPC for simple lem model runs.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from landlab import imshow_grid
from landlab.io.netcdf import read_netcdf, from_netcdf, to_netcdf

from landlab import RasterModelGrid, LinkStatus
from landlab.components import FlowAccumulator, ChiFinder
from landlab.grid.mappers import map_downwind_node_link_max_to_node

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']

# copy original run file and params
try:
    df_params = pd.read_csv('parameters.csv', index_col=0)[task_id]
    df_params.to_csv('../post_proc/%s/params_ID_%d.csv'%(base_output_path,ID), index=True)
except:
    print('could not find parameters.csv')

########## Load and basic plot
grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
iteration = int(files[-1].split('_')[-1][:-3])

# get parameter types right
for ind in df_params.index:
    try:
        df_params[ind] = float(df_params[ind])
    except ValueError:
        df_params[ind] = str(df_params[ind])
try:
    m = df_params['m']
except KeyError:
    m = 0.5
try: 
    n = df_params['n']
except KeyError:
    n = 1.0
try:
    bc = list(str(int(df_params['BCs'])))
except KeyError:
    bc = None
try:
    mg = from_netcdf(files[-1])
except KeyError:
    mg = read_netcdf(files[-1])
elev = mg.at_node['topographic__elevation']
bc_dict = {'4':mg.BC_NODE_IS_CLOSED, '1':mg.BC_NODE_IS_FIXED_VALUE}
if bc is not None:
    mg.set_status_at_node_on_edges(
            right=bc_dict[bc[0]],
            top=bc_dict[bc[1]],
            left=bc_dict[bc[2]],
            bottom=bc_dict[bc[3]],
    )       
else:
    mg.set_status_at_node_on_edges(
            right=mg.BC_NODE_IS_CLOSED,
            top=mg.BC_NODE_IS_CLOSED,
            left=mg.BC_NODE_IS_FIXED_VALUE,
            bottom=mg.BC_NODE_IS_CLOSED,
    )
# elevation
plt.figure(figsize=(8,6))
imshow_grid(mg, elev, cmap='gist_earth', colorbar_label='Elevation [m]', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID,iteration))
plt.savefig('../post_proc/%s/elev_ID_%d.png'%(base_output_path, ID))
plt.close()

##### steepness, curvature, and topographic index
fa = FlowAccumulator(mg, flow_director='D8', depression_finder='DepressionFinderAndRouter')
fa.run_one_step()

# calculate chi everywhere
cf = ChiFinder(mg, min_drainage_area=100*mg.dx**2, reference_concavity=m/n, reference_area=mg.dx**2)
cf.calculate_chi()

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

filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
to_netcdf(mg, filename, format="NETCDF4")

# relief change (nondimensional version)
# output_interval = int(files[1].split('_')[-1][:-3]) - int(files[0].split('_')[-1][:-3])
# dt_nd = output_interval*df_params['dt']/df_params['tg']
# relief_change = np.zeros(len(files))
# for i in range(1,len(files)):
#     grid = from_netcdf(files[i])
#     elev = grid.at_node['topographic__elevation']
#     relief_change[i] = np.mean(elev[grid.core_nodes])

# r_change = pd.DataFrame()
# r_change['r_nd'] = relief_change[:]/df_params['hg']
# r_change['drdt_nd'] = np.diff(r_change['r_nd'], prepend=0.0)/dt_nd
# r_change['t_nd'] = np.arange(len(files))*dt_nd
# r_change.to_csv('../post_proc/%s/relief_change_%d.csv'%(base_output_path, ID))
