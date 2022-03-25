"""
This script runs the StreamPowerModel with:
-- HydrologyEventVadoseStreamPower
-- FastscapeEroder
-- TaylorNonLinearDiffuser
-- RegolithConstantThickness

------
Parameters must be supplied in CSV file, 'parameters.csv'. The CSV file must
have a header with the number corresponding to the 'SLURM_ARRAY_TASK_ID' and
must contain at least the following columns:

ksat: Saturated hydraulic conductivity (m/s)
p: steady precip rate (m/s)
pet: potential evapotranspiration rate (m/s)
tr: mean storm duration (s)
tb: mean interstorm duration (s)
ds: mean storm depth (m)
b: regolith thickness (m)
n: drainable porosity (-)
Sawc: plant available water content (-)
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
E0: streampower incision coefficient

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
    PrecipitationDistribution,
    )
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import (
    HydrologyEventVadoseStreamPower,
    HydrologyEventVadoseThresholdStreamPower,
    RegolithConstantThickness,
    SchenkVadoseModel,
    )

#slurm info
task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

try:
    df_params = pandas.read_csv('parameters.csv', index_col=0)[task_id]
except FileNotFoundError:
    print("Supply a parameter file, 'parameters.csv' with column title equal to TASK_ID")

# pull values for this run
ksat = df_params['ksat']
p = df_params['p']
pet = df_params['pet']
Sawc = df_params['Sawc']
b = df_params['b']
n = df_params['n']
tr = df_params['tr']
tb = df_params['tb']
ds = df_params['ds']

K = df_params['K']
Ksp = K/p # precip rate from Q* goes in K
D = df_params['D']
U = df_params['U']
hg = df_params['hg']
Nz = df_params['Nz']

Th = df_params['Th']
Tg = df_params['Tg']
ksf = df_params['ksf']
dtg_max = df_params['dtg_max']

try:
    E0 = df_params['E0']
except KeyError:
    E0 = 0.0
try:
    Sc = df_params['Sc']
except KeyError:
    Sc = 0.0

output = {}
output["output_interval"] = df_params['output_interval']
output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = './data/stoch_sp_svm_'
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

    print("Using supplied initial grid")

except:
    print("Initial grid not present or could not be read. Initializing new grid.")
    Nx = df_params['Nx']
    v0 = df_params['v0']
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

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid,
                                porosity=n,
                                hydraulic_conductivity=ksat,
                                regularization_f=0.01,
                                recharge_rate=0.0,
                                courant_coefficient=0.9,
                                vn_coefficient = 0.9,
)
pdr = PrecipitationDistribution(grid,
                                mean_storm_duration=tr,
                                mean_interstorm_duration=tb,
                                mean_storm_depth=ds,
                                total_t=Th,
)
pdr.seed_generator(seedval=1235)
if Sc > 0:
    ld = TaylorNonLinearDiffuser(grid, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True)
else:
    ld = LinearDiffuser(grid, linear_diffusivity=D)

#initialize other models
svm = SchenkVadoseModel(potential_evapotranspiration_rate=pet,
                        available_water_content=Sawc,
                        profile_depth=b,
                        num_bins=int(Nz),
)
if E0 > 0.0:
    hm = HydrologyEventVadoseThresholdStreamPower(grid,
                                        precip_generator=pdr,
                                        groundwater_model=gdp,
                                        vadose_model=svm,
                                        sp_threshold=E0,
                                        sp_coefficient=Ksp
    )
else:
    hm = HydrologyEventVadoseStreamPower(grid,
                                        precip_generator=pdr,
                                        groundwater_model=gdp,
                                        vadose_model=svm,
    )
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
                        maximum_morphological_dt=dtg_max,
                        total_morphological_time=Tg,
                        verbose=False,
                        output_dict=output,
)

mdl.run_model()
