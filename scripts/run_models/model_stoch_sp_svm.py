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
#%%
import os
import glob
import numpy as np
import pandas

from landlab import RasterModelGrid
import xarray as xr
from DupuitLEM.io import load_grid_from_dataset, load_fields_from_dataset
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

#%%
try:
    #slurm info
    task_id = os.environ['SLURM_ARRAY_TASK_ID']
    ID = int(task_id)
    test_mode = False
except KeyError:
    print("In testing mode. Using parameters from first row of parameters.csv")
    task_id = '0'
    ID = 0
    test_mode = True

try:
    df_params = pandas.read_csv('parameters.csv', index_col=0)[task_id]
except FileNotFoundError:
    print("Supply a parameter file, 'parameters.csv' with column title equal to TASK_ID")

# get dtypes right
df_params = df_params.apply(lambda x: pandas.to_numeric(x, errors='coerce')).fillna(df_params)

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

# boundary conditions from a string
try:
    bc = list(str(df_params['BCs']))
except KeyError:
    bc = None

# try the arguments that might be present
dtg_max = df_params.get('dtg_max', None)
E0 = df_params.get('E0', 0.0)
Sc = df_params.get('Sc', 0.0)
n_sp = df_params.get('n_sp', 1.0)
precip_lapse_function = df_params.get('precip_lapse_function', None)
pet_lapse_function = df_params.get('pet_lapse_function', None)

if precip_lapse_function == 'linear':
    cutoff_elev = df_params.get('precip_lapse_cutoff_elev', 2000.0)
    precip_slope = df_params.get('precip_lapse_slope', 9e-12)
    print('Applying linear precipitation lapse function with slope %1.2e m/s per m and cutoff elevation %1.1f m'%(precip_slope, cutoff_elev))

    def precip_fun(precip, elev, cutoff_elev=cutoff_elev, precip_slope=precip_slope):
        """Calculate the predicted precipitation based on the fitted linear relationship with elevation.
        Parameters:
        precip (float): (m/s or m) Precipitation value at baselevel
        elev (array-like): (m) An array of mean elevation values for which to calculate the predicted precipitation.
        cutoff_elev (float): (m) elevation above which precipitation is assumed constant (i.e., the relationship with elevation breaks down). Default is 2000 m.
        precip_slope (float): (m/s per m) slope of the linear relationship between precipitation and elevation. Default is 9e-12 m/s per m (~250 mm/yr/km), based on the fitted model for the southeastern US subset of CAMELS.
        """
        zmean = min(np.mean(elev), cutoff_elev)
        return precip_slope * zmean + precip
elif precip_lapse_function is not None:
     print('precip_lapse_function %s not recognized. No precipitation lapse rate will be applied.'%precip_lapse_function)
     precip_fun = None
else:
    precip_fun = None    

if pet_lapse_function == 'linear':
    cutoff_elev = df_params.get('pet_lapse_cutoff_elev', 2000.0)
    pet_slope = df_params.get('pet_lapse_slope', -5e-12)
    pet_min = df_params.get('pet_lapse_min', 1e-9)
    print('Applying linear PET lapse function with slope %1.2e m/s per m, cutoff elevation %1.1f m, and minimum PET %1.2e m/s'%(pet_slope, cutoff_elev, pet_min))

    def pet_fun(pet, elev, cutoff_elev=cutoff_elev, pet_slope=pet_slope, pet_min=pet_min):
        """Calculate the predicted PET based on the fitted linear relationship with elevation.
        Parameters:
        pet (float): (m/s) PET value at baselevel
        elev (array-like): (m) An array of mean elevation values for which to calculate the predicted PET.
        cutoff_elev (float): (m) elevation above which PET is assumed constant (i.e., the relationship with elevation breaks down). Default is 2000 m.
        pet_slope (float): (m/s per m) slope of the linear relationship between PET and elevation. Default is -5e-12 m/s per m (~150 mm/yr/km), based on the fitted model for the southeastern US subset of CAMELS.
        pet_min (float): (m/s) Minimum PET value. Default is 1e-9 m/s (~30 mm/year).
        """
        zmean = min(np.mean(elev), cutoff_elev)
        return max(pet_slope * zmean + pet, pet_min)
elif pet_lapse_function is not None:
     print('pet_lapse_function %s not recognized. No PET lapse rate will be applied.'%pet_lapse_function)
     pet_fun = None
else: 
    pet_fun = None

# see if ksat is specified as a constant or a function of depth
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
output["base_output_path"] = './data/stoch_sp_svm_' if not test_mode else './test_stoch_sp_svm_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid
try:
    paths = glob.glob('*.nc')
    if len(paths) > 1:
        print("more than one grid available. Using last in list")
    ds = xr.open_dataset(paths[-1])
    grid = load_grid_from_dataset(ds)
    load_fields_from_dataset(ds, grid)

    # allow user to override BCs if specified, otherwise keep saved grid status.  #TODO: saved grid status no longer a thing - load from field status_at_node
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

    print("Using supplied initial grid")

except:
    print("Initial grid not present or could not be read. Initializing new grid.")
    Nx = df_params['Nx']
    try:
        Ny = df_params['Ny']
    except KeyError:
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

#%%

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
    print('Threshold E0>0 detected, using HydrologyEventVadoseThresholdStreamPower model!')
    print('Note: lapse functions are not currently implemented for the threshold stream power model, so any specified lapse functions will be ignored.')
else:
    hm = HydrologyEventVadoseStreamPower(grid,
                                        precip_generator=pdr,
                                        groundwater_model=gdp,
                                        vadose_model=svm,
                                        precip_lapse_function=precip_fun,
                                        pet_lapse_function=pet_fun,
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

# %%
