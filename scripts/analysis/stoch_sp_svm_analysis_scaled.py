"""
This script processes output from Dupuit Landscape Evolution Model (LEM) runs with 
stochastic stream power components. It performs the following analyses:

1. Load and visualize model grid data (elevation, aquifer base, water table)
2. Initialize and run hydrological models together (groundwater, precipitation, vadose zone)
3. Compute topographic metrics (slope, curvature, relief, roughness)
4. Analyze runoff generation and partitioning (recharge, baseflow, event runoff)
5. Perform baseflow separation using linear interpolation during rain events
6. Calculate flow quantiles and hydrologic partitioning ratios
7. Generate spatial maps of effective discharge, recharge, and water table dynamics
8. Classify saturation zones (never, variable, always saturated)
9. Compute relief change rates over simulation time
10. Export results to NetCDF grid files and CSV time series

Requires environment variables BASE_OUTPUT_FOLDER, SOURCE_RUN_ID and VARIANT_INDEX (see below).
Outputs spatial fields and scalar metrics for each model run ID.

SCALED-TOPOGRAPHY VERSION: a coevolved run is used as the base case. Its coevolved topography is 
stretched about the baselevel elevation so that the standard deviation of elevation changes by a 
factor s (s=1 is the unmodified coevolved topography), and the (spatially constant) permeable 
thickness b, with b0 = df_params['b'], is either
  'fixed': b = b0 for all s,
  'prop' : b = b0*s (permeable thickness increases with the scaled std of elevation), or
  'inv'  : b = b0/s (permeable thickness decreases with the scaled std of elevation).
A job post-processes ONE (original coevolved run, variant) pair. run_batch_glic_scaled.sh maps a flat
SLURM array over all runs and variants: task t = run*N + variant, with N = len(VARIANTS) (7 for the
default scales 0.75, 1.0, 1.25), so runs 0-26 need --array=0-188. All results go in one output folder,
named by f'{run}_v{variant}'.

Requires environment variables BASE_OUTPUT_FOLDER (post_proc output folder), SOURCE_RUN_ID (ID of the
original coevolved run, i.e. its column in parameters.csv) and VARIANT_INDEX, set by the batch script
which also runs this script in the original run's folder.
"""

#%%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
from numpy.lib.stride_tricks import sliding_window_view

from landlab import imshow_grid, RasterModelGrid, HexModelGrid, LinkStatus
import xarray as xr
from DupuitLEM.io import load_grid_from_dataset, load_fields_from_dataset, _find_last_non_nan
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    )
from landlab.grid.mappers import map_downwind_node_link_max_to_node
from DupuitLEM.auxiliary_models import HydrologyEventVadoseStreamPower, HydrologyEventVadoseThresholdStreamPower, SchenkVadoseModel
from DupuitLEM.grid_functions import bind_avg_exp_ksat, bind_avg_recip_ksat, get_link_hydraulic_conductivity
from DupuitLEM.io import initialize_output_dataset, write_output_step


def calc_mean_relief(grid):
    elev = grid.at_node['topographic__elevation']
    x = np.unique(grid.x_of_node)
    varline = np.zeros(len(x))
    
    for i, xi in enumerate(x):
        row = np.where(grid.x_of_node == xi)[0][1:-1]
        zrow = elev[row]
        varline[i] = np.var(zrow)
    
    return np.sqrt(np.nanmean(varline))

def calc_mean_roughness(grid, window_size=8):
    """use numpy.lib.stride_tricks to calculate the mean roughness"""

    z = grid.at_node['topographic__elevation'].reshape(grid.shape)
    # Create sliding windows of the elevation grid
    windows = sliding_window_view(z, (window_size, window_size))
    # Compute variance in each window
    var_field = np.var(windows, axis=(-2, -1))
    # Compute mean of the variance field
    mean_var = np.nanmean(var_field)
    # Return the square root (mean roughness)
    return np.sqrt(mean_var)


def run_recession_analysis(grid, df, plot=True, save_path=None):
    """nonlinear recession analysis using mm/hr discharge and linear regression
    on dQ/dt vs Qavg pointcloud.

    dQ/dt = -a Qavg ^ b
    
    returns:
    a: coefficient (from exp of linear regression intercept)
    b: exponent (from linear regression coefficient)
    k_eq: equivalent time constant (1/hr) from a, b, and mean recession discharge.
    """
    df = df.copy()
    A_tot = np.sum(grid.cell_area_at_node[grid.core_nodes])
    Q  = df['qs'].values / A_tot * 3600 * 1000   # mm/hr
    dt = df['dt'].values / 3600                   # hr  (assumes dt[i] = step i -> i+1)
    r  = df['r'].values

    Q1, Q2, dth = Q[:-1], Q[1:], dt[:-1]
    dQdt = -(Q2 - Q1) / dth
    with np.errstate(divide='ignore', invalid='ignore'):
        Qm = np.where(np.isclose(Q1, Q2), 0.5*(Q1+Q2), (Q1 - Q2) / np.log(Q1 / Q2))  # log-mean

    cond = (dQdt > 0) & (r[:-1] == 0) & (Q1 > 0) & (Q2 > 0)
    X = np.log(Qm[cond]).reshape(-1, 1)
    y = np.log(dQdt[cond])
    ols = LinearRegression().fit(X, y)
    a, b = np.exp(ols.intercept_), ols.coef_[0]

    Qref = np.exp(np.mean(np.log(Qm[cond])))       # geometric mean
    k_eq = a*(2-b)*Qref**(b-1) if b < 2 else np.nan  # Q/S turnover rate, 1/hr

    if plot:
        plt.figure()
        sc = plt.scatter(df['Qavg'][cond], df['-dQ/dt'][cond], alpha=0.3)
        q_fit = np.linspace(df.loc[cond, 'Qavg'].min(), df.loc[cond, 'Qavg'].max(), 200)
        X_fit = np.log(q_fit).reshape(-1, 1)
        plt.plot(q_fit, np.exp(ols.predict(X_fit)), 'k--', label='OLS')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')

        if save_path:
            plt.savefig(save_path)
            plt.close()

    a = np.exp(ols.intercept_)
    b = ols.coef_[0]

    return a, b, k_eq

def run_recession_analysis_SQ(grid, df, plot=True, save_path=None):

    df = df.copy()
    A_tot = np.sum(grid.cell_area_at_node[grid.core_nodes])
    S = df['S'].values/A_tot * 1000
    Q  = df['qs'].values / A_tot * 3600 * 1000   # mm/hr
    dt = df['dt'].values / 3600                   # hr  (assumes dt[i] = step i -> i+1)
    r  = df['r'].values

    Q1, Q2, dth = Q[:-1], Q[1:], dt[:-1]
    dQdt = -(Q2 - Q1) / dth
    cond = (dQdt > 0) & (r[:-1] == 0) & (Q1 > 0) & (Q2 > 0)
    cond_pad = np.append(cond, False)

    X = np.log(S[cond_pad]).reshape(-1, 1)
    y = np.log(Q[cond_pad])
    ols = LinearRegression().fit(X, y)
    k, n = np.exp(ols.intercept_), ols.coef_[0]

    Qref = np.exp(np.mean(np.log(S[cond_pad])))       # geometric mean
    k_eq = k**(1/n) * Qref**(1-1/n) # Q/S turnover rate, 1/hr

    if plot:
        plt.figure()
        sc = plt.scatter(S[cond_pad], Q[cond_pad], alpha=0.3)
        s_fit = np.linspace(S[cond_pad].min(), S[cond_pad].max(), 200)
        X_fit = np.log(s_fit).reshape(-1, 1)
        plt.plot(s_fit, np.exp(ols.predict(X_fit)), 'k--', label='OLS')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')

        if save_path:
            plt.savefig(save_path)
            plt.close()

    k = np.exp(ols.intercept_)
    n = ols.coef_[0]

    return k, n, k_eq


#%%
base_output_path = os.environ['BASE_OUTPUT_FOLDER']
task_id = os.environ['SOURCE_RUN_ID'] # column of parameters.csv for the original coevolved run
variant_index = int(os.environ['VARIANT_INDEX']) # index into VARIANTS
ID = f'{task_id}_v{variant_index}' # names all outputs; unique per (run, variant)

# variants applied to each base run. SCALES can be overridden with a comma-separated
# environment variable, e.g. SCALES=0.5,0.75,1.0,1.25,1.5 (then change N in the sbatch array).
# b0 is the run's own permeable thickness, df_params['b'], set below once parameters are loaded.
SCALES = [float(x) for x in os.environ.get('SCALES', '0.75,1.0,1.25').split(',')] # factors on std of elevation
B_MODES = ['fixed', 'prop', 'inv']
VARIANTS = []
for s_ in SCALES:
    for m_ in B_MODES:
        if s_ == 1.0 and m_ != 'fixed':
            continue # b is identical to 'fixed' when s=1
        VARIANTS.append((s_, m_))
if 'N_VARIANTS' in os.environ:
    assert int(os.environ['N_VARIANTS']) == len(VARIANTS), \
        'N_VARIANTS=%s but VARIANTS has %d entries'%(os.environ['N_VARIANTS'], len(VARIANTS))
scale, b_mode = VARIANTS[variant_index]
#%%

########## Load and basic plot
file = glob.glob('./data/*.nc')[0]
xr_ds = xr.open_dataset(file)

grid = load_grid_from_dataset(xr_ds)
load_fields_from_dataset(xr_ds, grid, last_non_nan=True)
index = _find_last_non_nan(xr_ds, -1)

elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt = grid.at_node['water_table__elevation']


########## Run hydrological model
# load parameters and save just this ID (useful because some runs in a group have been redone with diff parameters)

df_params = pd.read_csv('parameters.csv', index_col=0)[task_id]
df_params.to_csv(os.path.join('..', 'post_proc', base_output_path, f'params_ID_{ID}.csv'), index=True)

# get dtypes right
df_params = df_params.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(df_params)

ne = df_params['ne'] #drainable porosity [-]
b0 = df_params['b'] #characteristic depth of the coevolved run [m]
# permeable thickness for this variant: 'fixed' b0; 'prop' increases with scaled std; 'inv' decreases
b = {'fixed': b0, 'prop': b0*scale, 'inv': b0/scale}[b_mode]
print('source run %s: scale %.2f, b_mode %s, b0 = %.3f m, b = %.3f m'%(task_id, scale, b_mode, b0, b))
p = df_params['p'] #average precipitation rate [m/s]
tg = df_params['tg']
dtg = df_params['dtg']
hg = df_params['hg']
pet = df_params['pet']
na = df_params['na'] #plant available volumetric water content
tr = df_params['tr'] #mean storm duration [s]
tb = df_params['tb'] #mean interstorm duration [s]
ds = df_params['ds'] #mean storm depth [m]
T_h = 2000*(tr+tb) #20*df_params['Th'] #total hydrological time [s]
sat_cond = 0.025 # distance from surface (units of hg) for saturation
conc = 0.5 # reference concacvity for chi analysis
Nz = int(df_params['Nz']) # number of bins in vadose zone model

# boundary conditions from a string
try:
    bc = list(str(df_params['BCs']))
except KeyError:
    bc = None

# streampower threshold (so that time above threshold can be calculated here)
try:
    E0 = df_params['E0']
    K = df_params['K']
    Ksp = K/p
except KeyError:
    E0 = 0.0

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


# hydraulic conductivity
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
    
    elif ksat_type == 'aniso':
        try:
            krot = df_params['k_rot']
            kmax = df_params['kmax']
            kmin = kmax * 1/(df_params['k_ratio'])
        except KeyError:
            print('could not find parameters k_rot, k_ratio, and/or kmax for ksat_type %s'%ksat_type)
    
    else:
        print('Could not find ksat_type %s'%ksat_type)
        raise KeyError
except KeyError:
    ksat_type = None
    ksat = df_params['ksat']


#initialize grid
mg = RasterModelGrid(grid.shape, xy_spacing=grid.dx)
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
# scale topography about the baselevel (fixed-value boundary) elevation, so that the
# outlet elevation is unchanged and the std of elevation is multiplied by `scale`.
fixed = mg.status_at_node == mg.BC_NODE_IS_FIXED_VALUE
z0 = np.mean(elev[fixed]) if np.any(fixed) else np.min(elev)
elev = z0 + scale*(elev - z0)
std_orig = np.std(grid.at_node['topographic__elevation'][mg.core_nodes])
print('baselevel z0 = %.3f m, std of elevation %.3f -> %.3f m'%(z0, std_orig, np.std(elev[mg.core_nodes])))

z = mg.add_zeros('node', 'topographic__elevation')
z[:] = elev
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = elev - b # use the (constant) thickness for this variant, not from the grid.
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = np.clip(z0 + scale*(wt - z0), zb, z) # scaled initial water table, kept within the aquifer

# elevation
plt.figure(figsize=(8,6))
imshow_grid(mg, z, cmap='gist_earth', colorbar_label='Elevation [m]', grid_units=('m','m'))
plt.savefig(os.path.join('..', 'post_proc', base_output_path, f'elev_ID_{ID}.png'))
plt.close()

# record the variant
pd.DataFrame({'source_run': task_id, 'scale': scale, 'b_mode': b_mode, 'b0': b0, 'b': b,
              'z0': z0, 'std_orig': std_orig, 'std_scaled': np.std(z[mg.core_nodes])},
             index=[ID]).to_csv(os.path.join('..', 'post_proc', base_output_path, f'variant_ID_{ID}.csv'))

#initialize components
if isinstance(mg, RasterModelGrid):
    method = 'D8'
elif isinstance(mg, HexModelGrid):
    method = 'Steepest'
else:
    raise TypeError("grid should be Raster or Hex")

if ksat_type == 'aniso':
    # calculate link hydraulic conductivity
    rot = np.array([[np.cos(krot), -np.sin(krot)],[np.sin(krot), np.cos(krot)]])
    K_principal = np.array([[kmax,0.0],[0.0, kmin]])
    K = rot @ K_principal @ inv(rot)
    ksat = get_link_hydraulic_conductivity(mg, K)

gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=ne,
                                  hydraulic_conductivity=ksat,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.05,
                                  vn_coefficient = 0.05,
                                  #callback_fun = write_SQ,
                                  )
pdr = PrecipitationDistribution(mg, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=T_h)
pdr.seed_generator(seedval=2)
svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                 available_water_content=na,
                 profile_depth=b,
                 num_bins=Nz,
                 )
svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
if E0 > 0.0:
    hm = HydrologyEventVadoseThresholdStreamPower(
        mg,
        precip_generator=pdr,
        groundwater_model=gdp,
        sp_threshold=E0,
        sp_coefficient=Ksp,
        vadose_model=svm,
        precip_lapse_function=precip_fun,
        pet_lapse_function=pet_fun,
        )
    print('Threshold E0>0 detected, using HydrologyEventVadoseThresholdStreamPower model!')
else: 
    hm = HydrologyEventVadoseStreamPower(
                                        mg,
                                        precip_generator=pdr,
                                        groundwater_model=gdp,
                                        vadose_model=svm,
                                        precip_lapse_function=precip_fun,
                                        pet_lapse_function=pet_fun,
                                        )

#run model (spinup to be safe)
hm.run_step()

f = open('../post_proc/%s/dt_qs_s_%s.csv'%(base_output_path, ID), 'w')
def write_SQ(grid, r, dt, file=f):
    cores = grid.core_nodes
    h = grid.at_node["aquifer__thickness"]
    wt = grid.at_node["water_table__elevation"]
    z = grid.at_node["topographic__elevation"]
    sat = (z-wt) < sat_cond*hg
    qs = grid.at_node["surface_water__specific_discharge"]
    area = grid.cell_area_at_node

    storage = np.sum(ne*h[cores]*area[cores])
    qs_tot = np.sum(qs[cores]*area[cores])
    sat_nodes = np.sum(sat[cores])
    r_tot = np.sum(r[cores]*area[cores])

    file.write('%f, %f, %f, %f, %f\n'%(dt, r_tot, qs_tot, storage, sat_nodes))
gdp.callback_fun = write_SQ

hm.run_step(record_state=True)
f.close()

##########  Analysis

#dataframe for output
df_output = {}

##### slope and curvature
S8 = mg.add_zeros('node', 'slope_D8')
S4 = mg.add_zeros('node', 'slope_D4')
curvature = mg.add_zeros('node', 'curvature')

#D8 slope is the absolute value of D8 gradient associated with
#flow direction. Same as FastscapeEroder. curvature is divergence of gradient.
#Same as LinearDiffuser.
dzdx_D8 = mg.calc_grad_at_d8(elev)
dzdx_D4 = mg.calc_grad_at_link(elev)
dzdx_D4[mg.status_at_link == LinkStatus.INACTIVE] = 0.0
S8[:] = abs(dzdx_D8[mg.at_node['flow__link_to_receiver_node']])
S4[:] = map_downwind_node_link_max_to_node(mg, dzdx_D4)

curvature[:] = mg.calc_flux_div_at_node(dzdx_D4)

roughness_window_size = 8
df_output['mean_elev'] = np.mean(elev[mg.core_nodes])
df_output['max_elev'] = np.max(elev[mg.core_nodes])
df_output['mean_rough'] = calc_mean_roughness(mg, window_size=roughness_window_size)
df_output['mean_rel'] = calc_mean_relief(mg)
df_output['elev_std'] = np.std(elev[mg.core_nodes])
df_output['mean_slope'] = S8[mg.core_nodes].mean()
df_output['median_slope'] = np.median(S8[mg.core_nodes])
df_output['mean_curv'] = curvature[mg.core_nodes].mean()


######## Runoff generation
df_output['cum_precip'] = hm.cum_precip
df_output['cum_recharge'] = hm.cum_recharge
df_output['cum_runoff'] = hm.cum_runoff
df_output['cum_gw_export'] = hm.cum_gw_export
df_output['cum_pet'] = hm.cum_pet

"""ratio of total recharge to total precipitation, averaged over space and time.
this accounts for time varying recharge with precipitation rate, unsat
storage and ET, as well as spatially variable recharge with water table depth.
"""
df_output['recharge_efficiency'] = hm.cum_recharge / hm.cum_precip
df_output['(P-Q-Qgw)/P'] = (hm.cum_precip - hm.cum_runoff - hm.cum_gw_export)/hm.cum_precip
df_output['Q/P'] = hm.cum_runoff/hm.cum_precip

#load the full storage discharge dataset that was just generated
timeseries_path = os.path.join('..', 'post_proc', base_output_path, f'dt_qs_s_{ID}.csv')
df = pd.read_csv(timeseries_path, sep=',',header=None, names=['dt','r', 'qs', 'S', 'sat_nodes'])

# remove the first row (test row written by init of gdp)
df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)
# make dimensionless timeseries
Atot = np.sum(mg.cell_area_at_node[mg.core_nodes])
df['qs_star'] = df['qs']/(p*Atot)
df['S_star'] = df['S']/(b*ne*Atot)

# find times with recharge. Note in df qs and S are at the end of the timestep.
# i is at the beginning of the timestep.
df['t'] = np.cumsum(df['dt'])
is_r = df['r'] > 0.0
pre_r = np.where(np.diff(is_r*1)>0.0)[0] #last data point before recharge
end_r = np.where(np.diff(is_r*1)<0.0)[0][1:] #last data point where there is recharge

# make a linear-type baseflow separation, where baseflow during recharge increases
# linearly from its initial qs before recharge to its final qs after recharge.
qb = df['qs'].copy()
q = df['qs']
t = df['t']
for i in range(len(end_r)):
    j = pre_r[i]
    k = end_r[i]
    slope = (q[k+1] - q[j])/(t[k+1]-t[j])
    qb[j+1:k+1] = q[j]+slope*(t[j+1:k+1] - t[j])

df['qb'] = qb
qe = df['qs'] - df['qb']

# integrate to find total values. trapezoidal for the properties that
# change dynamically, and simple rectangular for i bc we know rain varies in this way.
df_output['qe_tot'] = np.sum(df['dt'] * qe)
df_output['qb_tot'] = np.sum(df['dt'] * qb)
df_output['qs_tot'] = np.sum(df['dt'] * df['qs'])
df_output['r_tot'] = np.sum(df['dt'] * df['r']) # recharge

# L'vovich partitioning: wetting on precipitation
df_output['W/P'] = (df_output['cum_precip'] - df_output['qe_tot'])/df_output['cum_precip']
df_output['Qb/W'] = df_output['qb_tot']/(df_output['cum_precip'] - df_output['qe_tot'])
df_output['Qb/Q'] = df_output['qb_tot']/df_output['qs_tot'] #baseflow index
# df_output['RR'] = qe_tot/r_tot #runoff ratio

# flow quantiles
df_output['Q_50'] = np.quantile(df['qs'], 0.5, weights=df['dt'], method="inverted_cdf")
df_output['Q_95'] = np.quantile(df['qs'], 0.95, weights=df['dt'], method="inverted_cdf")
df_output['Q_90'] = np.quantile(df['qs'], 0.90, weights=df['dt'], method="inverted_cdf")
df_output['Q_5'] = np.quantile(df['qs'], 0.05, weights=df['dt'], method="inverted_cdf")
df_output['Q_10'] = np.quantile(df['qs'], 0.10, weights=df['dt'], method="inverted_cdf")


df['S_mm'] = df['S']/Atot * 1000
df_output['S_max_mm'] = df['S_mm'].max()
df_output['S_range_mm'] = df['S_mm'].max() - df['S_mm'].min()
df_output['S_mean_mm'] = np.average(df['S_mm'].values, weights=df['dt'])
df_output['S_std_mm'] = np.sqrt(np.average((df['S_mm'] - df_output['S_mean_mm'])**2, weights=df['dt']))
df_output['S_5_mm'] = np.quantile(df['S_mm'].values, 0.05, weights=df['dt'], method="inverted_cdf")
df_output['S_95_mm'] = np.quantile(df['S_mm'].values, 0.95, weights=df['dt'], method="inverted_cdf")
df_output['S_5_95_mm'] = df_output['S_95_mm'] - df_output['S_5_mm']

# recession
rec_a, rec_b, k_eq = run_recession_analysis(mg, df, plot=False) # do not overwrite b (permeable thickness)
df_output['rec_a'] = rec_a
df_output['rec_b'] = rec_b
df_output['rec_k_eq_1/hr'] = k_eq

rec_k, rec_n, k_eq_S = run_recession_analysis_SQ(mg, df, plot=False)
df_output['rec_n'] = rec_n
df_output['rec_k'] = rec_k
df_output['rec_k_eq_S_1/hr'] = k_eq_S


df_output['q_avg_mm/hr'] = np.average(df['qs'].values, weights=df['dt']) / Atot * 1000 * 3600


###### spatial runoff related quantities

# effective Qstar
intensity = hm.intensity[:-1]

qstar = mg.add_zeros('node', 'Q_star')
q_eff = mg.at_node["surface_water_effective__discharge"]
qstar[:] = q_eff/(p*mg.at_node['drainage_area'])

# mean and variance of water table
wt_all = hm.wt_all[1:,:]
base_all = np.ones(wt_all.shape)*mg.at_node['aquifer_base__elevation']
elev_all = np.ones(wt_all.shape)*mg.at_node['topographic__elevation']
wtrel_all = np.zeros(wt_all.shape)
wtrel_all[:, mg.core_nodes] = (wt_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])/(elev_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])

wtrel_05 = mg.add_zeros('node', 'wtrel_05')
wtrel_95 = mg.add_zeros('node', 'wtrel_95')
Srel = np.mean(wtrel_all, axis=1) #relative storage timeseries
quantiles = np.quantile(Srel, [0.05, 0.95])
quan = [(np.abs(Srel - i)).argmin() for i in quantiles]
wtrel_05[:] = wtrel_all[quan[0], :]
wtrel_95[:] = wtrel_all[quan[1], :]

# water table and saturation at end of storm and interstorm
sat_all = (elev_all-wt_all) < sat_cond*hg
sat_end_storm = np.mean(sat_all[intensity>0,:], axis=0)
sat_end_interstorm = np.mean(sat_all[intensity==0.0,:], axis=0)

# classify zones of saturation
sat_class = mg.add_zeros('node', 'saturation_class')
sat_never = np.logical_and(sat_end_storm < 0.05, sat_end_interstorm < 0.05)
sat_always = np.logical_and(sat_end_interstorm > 0.95, sat_end_storm > 0.95)
sat_variable = ~np.logical_or(sat_never, sat_always)
sat_class[sat_never] = 0
sat_class[sat_variable] = 1
sat_class[sat_always] = 2
df_output['sat_never'] = np.sum(sat_never[mg.core_nodes])/mg.number_of_core_nodes
df_output['sat_variable'] = np.sum(sat_variable[mg.core_nodes])/mg.number_of_core_nodes
df_output['sat_always'] = np.sum(sat_always[mg.core_nodes])/mg.number_of_core_nodes

#%% ####### save things

output_fields = [
    "at_node:topographic__elevation",
    "at_node:aquifer_base__elevation",
    "at_node:saturation_class",
    "at_node:slope_D8",
    "at_node:slope_D4",
    "at_node:curvature",
    "at_node:wtrel_05",
    "at_node:wtrel_95",
    "at_node:Q_star",
    ]

base_file_name = os.path.join('..', 'post_proc', base_output_path)
filename = os.path.join(base_file_name, f'grid_{ID}.nc')
output = {'output_fields': output_fields, 'run_id': ID, 'output_interval': 1}

ds_out = initialize_output_dataset(mg, output, [index])
write_output_step(ds_out, mg, output, 0)
ds_out.to_netcdf(filename, format="NETCDF4")

# record the variant alongside the results
df_output['source_run'] = task_id
df_output['scale'] = scale
df_output['b_mode'] = b_mode
df_output['b0'] = b0
df_output['b'] = b
df_output = pd.DataFrame.from_dict(df_output, orient='index', columns=[ID])
df_output.to_csv(os.path.join(base_file_name, f'output_ID_{ID}.csv'))
df.to_csv(os.path.join(base_file_name, f'q_s_dt_ID_{ID}.csv'))