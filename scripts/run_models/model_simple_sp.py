
import os
import numpy as np
from tqdm import tqdm
import pandas

from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    FastscapeEroder,
    LinearDiffuser,
    TaylorNonLinearDiffuser,
    FlowDirectorD8,
    FlowDirectorSteepest,
    LakeMapperBarnes,
    DepressionFinderAndRouter,
    )
from landlab.io.netcdf import to_netcdf

#slurm info
task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

try:
    df_params = pandas.read_csv('parameters.csv', index_col=0)[task_id]
    # df = pd.read_csv('df_params_1d_%d.csv'%ID, index_col=0)
except FileNotFoundError:
    print("Supply a parameter file, 'parameters.csv' with column title equal to TASK_ID")

for ind in df_params.index:
    try:
        df_params[ind] = float(df_params[ind])
    except ValueError:
        df_params[ind] = str(df_params[ind])

try:
    Nx = df_params['Nx']
    Ny = df_params['Ny']
except KeyError:
    Nx = df_params['Nx']
    Ny = df_params['Nx']
T = df_params['T']
dt = df_params['dt']
v0 = df_params['v0']
D = df_params['D']
K = df_params['K']
U = df_params['U']
m = df_params['m']
n = df_params['n']
try:
    Sc = df_params['Sc']
except KeyError:
    Sc = 0.0
try:
    bc = list(str(df_params['BCs']))
except KeyError:
    bc = None
try:
    output_interval = df_params['output_interval']
    save_transients = True
except KeyError:
    output_interval = None
    save_transients = False
try:
    r_condition = df_params['r_condition']
except KeyError:
    r_condition = 0.0
try:
    routing_method = df_params['routing_method']
except ValueError:
    routing_method = 'D8'
save_directory = './data'

N = int(T//dt)

np.random.seed(12345)
grid = RasterModelGrid((Ny, Nx), xy_spacing=v0)
bc_dict = {'4':grid.BC_NODE_IS_CLOSED, '1':grid.BC_NODE_IS_FIXED_VALUE}
if bc is not None:
    grid.set_status_at_node_on_edges(
            right=bc_dict[bc[0]],
            top=bc_dict[bc[1]],
            left=bc_dict[bc[2]],
            bottom=bc_dict[bc[3]],
    )       
else:
    grid.set_status_at_node_on_edges(
            right=grid.BC_NODE_IS_CLOSED,
            top=grid.BC_NODE_IS_CLOSED,
            left=grid.BC_NODE_IS_FIXED_VALUE,
            bottom=grid.BC_NODE_IS_CLOSED,
    )
z = grid.add_zeros('node', 'topographic__elevation')
z[grid.core_nodes] = 0.2*np.random.rand(len(grid.core_nodes))

# depression_finder='DepressionFinderAndRouter', routing='D8'
# fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8', depression_finder='LakeMapperBarnes', method='D8')
if routing_method == "D8":
    fd = FlowDirectorD8(grid)
elif routing_method == "Steepest":
    fd = FlowDirectorSteepest(grid)
else:
    raise ValueError("routing_method must be either D8 or Steepest.")
fa = FlowAccumulator(
    grid,
    surface="topographic__elevation",
    flow_director=fd,
)
lmb = LakeMapperBarnes(
    grid,
    method=routing_method,
    fill_flat=False,
    surface="topographic__elevation",
    fill_surface="topographic__elevation",
    redirect_flow_steepest_descent=False,
    reaccumulate_flow=False,
    track_lakes=False,
    ignore_overfill=True,
)
dfr = DepressionFinderAndRouter(grid)
if Sc > 0:
    ld = TaylorNonLinearDiffuser(grid, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True)
else:
    ld = LinearDiffuser(grid, linear_diffusivity=D)
sp = FastscapeEroder(grid, K_sp=K, m_sp=m, n_sp=n)


relief = np.zeros(N)

for i in tqdm(range(N), desc="Completion"):

    # baselevel change
    z[grid.core_nodes] += U*dt

    # diffusion
    ld.run_one_step(dt)

    # flow directions and incision
    dfr._find_pits()
    if dfr._number_of_pits > 0:
        lmb.run_one_step()
    fa.run_one_step()
    sp.run_one_step(dt)

    # save output
    if save_transients:
        if i%output_interval==0:
            filename = os.path.join(save_directory, 'simple_sp_%d_grid_%d.nc'%(ID,i))
            to_netcdf(grid, filename)

    # check relief condition
    relief[i] = np.mean(z[grid.core_nodes])
    if r_condition > 0.0 and i > 100:
        if np.mean(np.abs(np.diff(relief[i-50:i])/dt)) < r_condition:
            filename = os.path.join(save_directory, 'simple_sp_%d_grid_%d.nc'%(ID,i))
            to_netcdf(grid, filename)
            break

# save relief
np.savetxt('relief_%d.csv'%ID, relief, delimiter=',')
