"""
This script runs the StreamPowerModel with:
-- HydrologySteadyStreamPower
-- FastscapeEroder
-- TaylorNonLinearDiffuser
-- RegolithConstantThickness

------
Parameters must be supplied in CSV file, 'parameters.csv'. CSV file must
contain at least the following columns:

ksat: Saturated hydraulic conductivity (m/s)
p: steady precip rate (m/s)
b: regolith thickness (m)
n: drainable porosity (-)
K: streampower incision coefficient (1/s)
D: hillslope diffusivity (m2/s)
U: uplift rate (m/s)
hg: geomorphic height scale (m) (only for scaling intial surface roughness)
Sc: critical hillslope slope (-)
Th: hydrological simulation time (s)
Tg: total geomorphic time (s)
ksf: scaling factor between hydrological time and geomorphic time (-)
output_interval: how frequently grid will be saved (timesteps)
v0: grid spacing (m) (if grid not supplied)
Nx: number of nodes in x and y dimension (if grid not supplied)

Optional:
RE: recharge efficiency (-) If supplied, recharge rate is p*RE.

-------
Starting grid can be supplied as NETCDF4 created with landlab, 'grid.nc' with
fields:
- "topographic__elevation"
- "water_table__elevation"
- "aquifer_base__elevation"
at node.
"""
import os
import numpy as np
import pandas

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    TaylorNonLinearDiffuser,
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
    df_params = pandas.read_csv('parameters.csv')
    # df = pd.read_csv('df_params_1d_%d.csv'%ID, index_col=0)
except FileNotFoundError:
    print("Supply a parameter file, 'parameters.csv'")

# pull values for this run
ksat = df_params['ksat'][ID]
p = df_params['p'][ID]
b = df_params['b'][ID]
n = df_params['n'][ID]

K = df_params['K'][ID]
Ksp = K/p # precip rate from Q* goes in K
D = df_params['D'][ID]
U = df_params['U'][ID]
hg = df_params['hg'][ID]
Sc = df_params['Sc'][ID]

Th = df_params['Th'][ID]
Tg = df_params['Tg'][ID]
ksf = df_params['ksf'][ID]
try:
    RE = df_params['RE'][ID]
except KeyError:
    print("no recharge efficiency 'RE' provided. Using RE = 1.0")
    RE = 1.0

output = {}
output["output_interval"] = df_params['output_interval'][ID]
output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = './data/steady_sp_nld_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid
try:
    mg = from_netcdf('grid.nc')
    z = mg.at_node['topographic__elevation']
    zb = mg.at_node['aquifer_base__elevation']
    zwt = mg.at_node['water_table__elevation']
    print("Using supplied initial grid")

    grid = RasterModelGrid(mg.shape, xy_spacing=mg.dx)
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
    Nx = df_params['Nx'][ID]
    v0 = df_params['v0'][ID]
    np.random.seed(12345)
    grid = RasterModelGrid((Nx, Nx), xy_spacing=v0)
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
        porosity=n,
        hydraulic_conductivity=ksat,
        regularization_f=0.01,
        recharge_rate=p*RE,
        courant_coefficient=0.1,
        vn_coefficient = 0.1,
)
ld = TaylorNonLinearDiffuser(grid, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True)

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
