"""
This script runs the StreamPowerModel with:
-- HydrologyEventVadoseStreamPower/HydrologyEventVadoseThresholdStreamPower
-- FastscapeEroder
-- TaylorNonLinearDiffuser/LinearDiffuser
-- RegolithConstantThickness

------
Parameters must be supplied in CSV file, 'parameters.csv'. The CSV file must
have a header with the number corresponding to the 'SLURM_ARRAY_TASK_ID' and
must contain at least the following columns:

ksat: saturated hydraulic conductivity (m/s) (if not ksat_type)
p: steady precip rate (m/s)
pet: potential evapotranspiration rate (m/s)
tr: mean storm duration (s)
tb: mean interstorm duration (s)
ds: mean storm depth (m)
b: regolith thickness (m)
ne: drainable porosity (-)
na: plant available water content (-)
K: streampower incision coefficient (1/s)
n_sp: slope coefficient in streampower model (-)
D: hillslope diffusivity (m2/s)
U: uplift rate (m/s)
hg: geomorphic height scale (m) (only for scaling initial surface roughness)
Th: hydrological simulation time (s)
Tg: total geomorphic time (s)
ksf: scaling factor between hydrological time and geomorphic time (-)
output_interval: how frequently grid will be saved (timesteps)
v0: grid spacing (m) (if grid not supplied)
Nx: number of nodes in x dimension (if grid not supplied)
Ny: number of nodes in y dimension (if grid not supplied. If not supplied, grid is Nx,Nx)

Optional:
E0: streampower incision threshold (m/s)
Sc: critical hillslope slope (-)
BCs: number/string that tells the model how to set the boundary conditions.
    1 indicates a fixed value boundary, 4 indicates a closed boundary. They
    are ordered RightTopLeftBottom, e.g. 4414, 1111.
    Default: 4414
ksat_type: a way to specify that the hydraulic conductivity changes with depth
    according to one of the functions in grid_funcs. Options at the moment are
    'exp' for exponential form, and 'recip' for reciprocal form. Each has 
    additional arguments:
    'exp':
        ksurface: (ks) surface hydraulic conductivity
        kdepth: (k0) a lower conductivity
        kdecay: (dk) decline rate coefficient
    'recip':
        ksurface: (ks) surface hydraulic conductivity
        kdecay: (dk) decline rate coefficient


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
from DupuitLEM.grid_functions import (
    bind_avg_exp_ksat, 
    bind_avg_recip_ksat,
)

#slurm info
task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

try:
    df_params = pandas.read_csv('parameters.csv', index_col=0)[task_id]
except FileNotFoundError:
    print("Supply a parameter file, 'parameters.csv' with column title equal to TASK_ID")

# get dtypes right
for ind in df_params.index:
    try:
        df_params[ind] = float(df_params[ind])
    except ValueError:
        df_params[ind] = str(df_params[ind])

# pull values for this run
p = df_params['p']
pet = df_params['pet']
b = df_params['b']
ne = df_params['ne']
na = df_params['na']
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

# try the arguments that might be present
try:
    E0 = df_params['E0']
except KeyError:
    E0 = 0.0
try:
    Sc = df_params['Sc']
except KeyError:
    Sc = 0.0
try:
    n_sp = df_params['n_sp']
except KeyError:
    n_sp = 1.0
try:
    bc = list(str(df_params['BCs']))
except KeyError:
    bc = None
try:
    ksat_type = df_params['ksat_type']

    if ksat_type == 'recip':
        try:
            ks = df_params['ksurface']
            d = df_params['kdecay']

            ksat = bind_avg_recip_ksat(ks, d)
        except KeyError:
            print('could not find parameters ksurface and/or kdecay for ksat_type %s'%ksat_type)

    elif ksat_type == 'exp':
        try:
            ks = df_params['ksurface']
            k0  = df_params['kdepth']
            dk = df_params['kdecay']

            ksat = bind_avg_exp_ksat(ks, k0, dk)
        except KeyError:
            print('could not find parameters ksurface, kdepth, and/or kdecay for ksat_type %s'%ksat_type)
    else:
        print('Could not find ksat_type %s'%ksat_type)
        raise KeyError
except KeyError:
    ksat = df_params['ksat']

# make output dictionary
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

    print("Using supplied initial grid")

except:
    print("Initial grid not present or could not be read. Initializing new grid.")
    try:
        Nx = df_params['Nx']
        Ny = df_params['Ny']
    except KeyError:
        Nx = df_params['Nx']
        Ny = df_params['Nx']

    v0 = df_params['v0']
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
    elev = grid.add_zeros('node', 'topographic__elevation')
    elev[:] = b + 0.1*hg*np.random.rand(len(elev))
    base = grid.add_zeros('node', 'aquifer_base__elevation')
    base[:] = elev - b
    wt = grid.add_zeros('node', 'water_table__elevation')
    wt[:] = elev.copy()

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid,
                                porosity=ne,
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
                        available_water_content=na,
                        profile_depth=b,
                        num_bins=int(Nz),
)
svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
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
# m_sp is set to 1, but in our forumulation, the norm discharge
# field results in the form E = K Q* A^1/2 S^n_sp
sp = FastscapeEroder(grid,
                    K_sp=Ksp,
                    m_sp=1,
                    n_sp=n_sp,
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
