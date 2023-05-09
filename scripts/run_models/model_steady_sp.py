"""
This script runs the StreamPowerModel with:
-- HydrologySteadyStreamPower
-- FastscapeEroder
-- TaylorNonLinearDiffuser/LinearDiffuser
-- RegolithConstantThickness

------
Parameters must be supplied in CSV file, 'parameters.csv'. The CSV file must
have a header with the number corresponding to the 'SLURM_ARRAY_TASK_ID' and
must contain at least the following columns:

ksat: Saturated hydraulic conductivity (m/s)
p: steady precip rate (m/s)
b: regolith thickness (m)
ne: drainable porosity (-)
K: streampower incision coefficient (1/s)
D: hillslope diffusivity (m2/s)
U: uplift rate (m/s)
hg: geomorphic height scale (m) (only for scaling initial surface roughness)
Sc: critical hillslope slope (-)
Th: hydrological simulation time (s)
Tg: total geomorphic time (s)
ksf: scaling factor between hydrological time and geomorphic time (-)
output_interval: how frequently grid will be saved (timesteps)
v0: grid spacing (m) (if grid not supplied)
Nx: number of nodes in x and y dimension (if grid not supplied)

Optional:
RE: recharge efficiency (-) If supplied, recharge rate is p*RE.
E0: streampower incision threshold.
Sc: critical hillslope slope (-)
BCs: number/string that tells the model how to set the boundary conditions.
    1 indicates a fixed value boundary, 4 indicates a closed boundary. They
    are ordered RightTopLeftBottom, e.g. 4414, 1111.
    Default: 4414


-------
Starting grid can be supplied as NETCDF4 created with landlab, 'grid.nc' with
fields:
- "topographic__elevation"
- "water_table__elevation"
- "aquifer_base__elevation"
at node.


6 Dec 2021
"""

import os
import glob
import numpy as np
import pandas

from landlab import RasterModelGrid
from landlab.io.netcdf import from_netcdf
from landlab.components import (
    GroundwaterDupuitPercolator,
    TaylorNonLinearDiffuser,
    LinearDiffuser,
    FastscapeEroder,
    )
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import (
    HydrologySteadyStreamPower,
    RegolithConstantThickness,
    )

#slurm info
task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

try:
    df_params = pandas.read_csv('parameters.csv', index_col=0)[task_id]
    # df = pd.read_csv('df_params_1d_%d.csv'%ID, index_col=0)
except FileNotFoundError:
    print("Supply a parameter file, 'parameters.csv' with column title equal to TASK_ID")

# pull values for this run
ksat = df_params['ksat']
p = df_params['p']
b = df_params['b']
ne = df_params['ne']

K = df_params['K']
Ksp = K/p # precip rate from Q* goes in K
D = df_params['D']
U = df_params['U']
hg = df_params['hg']

Th = df_params['Th']
Tg = df_params['Tg']
ksf = df_params['ksf']
try:
    RE = df_params['RE']
except KeyError:
    print("no recharge efficiency 'RE' provided. Using RE = 1.0")
    RE = 1.0
try:
    E0 = df_params['E0']
except KeyError:
    E0 = 0.0
try:
    Sc = df_params['Sc']
except KeyError:
    Sc = 0.0
try:
    bc = list(str(df_params['BCs']))
except KeyError:
    bc = None


output = {}
output["output_interval"] = df_params['output_interval']
output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = './data/steady_sp_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid
try:
    paths = glob.glob('*.nc')
    if len(paths) > 1:
        print("more than one grid available. Using last in list")
    mg = from_netcdf(paths[-1])
    z = mg.at_node['topographic__elevation']
    zb = mg.at_node['aquifer_base__elevation']
    zwt = mg.at_node['water_table__elevation']
    print("Using supplied initial grid")

    grid = RasterModelGrid(mg.shape, xy_spacing=mg.dx)
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
    elev = grid.add_zeros('node', 'topographic__elevation')
    elev[:] = z.copy()
    base = grid.add_zeros('node', 'aquifer_base__elevation')
    base[:] = zb.copy()
    wt = grid.add_zeros('node', 'water_table__elevation')
    wt[:] = zwt.copy()

except:
    print("Initial grid not present or could not be read. Initializing new grid.")
    Nx = df_params['Nx']
    v0 = df_params['v0']
    np.random.seed(12345)
    grid = RasterModelGrid((Nx, Nx), xy_spacing=v0)
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
    elev = grid.add_zeros('node', 'topographic__elevation')
    elev[:] = b + 0.1*hg*np.random.rand(len(elev))
    base = grid.add_zeros('node', 'aquifer_base__elevation')
    wt = grid.add_zeros('node', 'water_table__elevation')
    wt[:] = elev.copy()

#initialize components
gdp = GroundwaterDupuitPercolator(grid,
        porosity=ne,
        hydraulic_conductivity=ksat,
        regularization_f=0.01,
        recharge_rate=p*RE,
        courant_coefficient=0.1,
        vn_coefficient = 0.1,
)
if Sc > 0.0:
    ld = TaylorNonLinearDiffuser(grid, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True)
else:
    ld = LinearDiffuser(grid, linear_diffusivity=D)

hm = HydrologySteadyStreamPower(
        grid,
        groundwater_model=gdp,
        hydrological_timestep=Th,
)

# surface_water_area_norm__discharge (Q/sqrt(A)) = Q* p v0 sqrt(a)
sp = FastscapeEroder(grid,
        K_sp=Ksp,
        m_sp=1,
        n_sp=1,
        discharge_field="surface_water_area_norm__discharge",
        threshold_sp=E0,
)
rm = RegolithConstantThickness(grid, equilibrium_depth=b, uplift_rate=U)

mdl = StreamPowerModel(grid,
        hydrology_model=hm,
        diffusion_model=ld,
        erosion_model=sp,
        regolith_model=rm,
        morphologic_scaling_factor=ksf,
        total_morphological_time=Tg,
        verbose=True,
        output_dict=output,
)

mdl.run_model()
