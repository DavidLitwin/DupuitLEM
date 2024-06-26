"""
Generate parameters for StreamPowerModel with
-- HydrologySteadyStreamPower
-- FastscapeEroder
-- LinearDiffuser or TaylorNonLinearDiffuser
-- RegolithConstantThickness

This script generates the uses a 1D hillslope model to determine a
recharge_efficiency 'RE' when varying rho and sigma, and then assumes that
this recharge rate will be applied to a steady recharge DupuitLEM model. This
allows us to compare the steady and stochastic models that are close to equivalent.

alpha = hg / lg
gamma = (b ksat hg) / (p lg^2)
Hi = (ksat hg^2) / (p lg^2)
sigma = (b n) / (p (tr + tb))
rho = tr / (tr + tb)
ai = (pet tb) / (p (tr + tb))
theta = E0 tg / hg
phi = na / ne

6 Dec 2021
"""

import os
import numpy as np
from itertools import product
import pandas as pd

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    )

from DupuitLEM.auxiliary_models import (
    HydrologyEventVadoseStreamPower,
    SchenkVadoseModel,
    )
#print("modules imported")

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#dim equations
def K_fun(v0, lg, tg):
    return np.sqrt(lg)/(np.sqrt(v0)*tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

def b_fun(hg, gam, hi):
    return (hg*gam)/hi

def ksat_fun(p, hg, lg, hi):
    return (lg**2*p*hi)/hg**2

def E0_fun(theta, hg, tg):
    return theta*(hg/tg)

# additional for recharge estimation:
def ds_fun(hg, ne, gam, sigma, hi):
    return (hg*ne*gam)/(hi*sigma)

def tr_fun(hg, p, ne, gam, sigma, hi, rho):
    return (hg*ne*gam*rho)/(p*sigma*hi)

def tb_fun(hg, p, ne, gam, sigma, hi, rho):
    return (hg*ne*gam)*(1-rho)/(p*sigma*hi)

def pet_fun(p, rho, ai):
    return (ai*p)/(1-rho)

def calc_z(x, Sc, U, D):
    """Nonlinear diffusion elevation profile"""
    t1 = np.sqrt(D**2 + (2*U*x/Sc)**2)
    t2 = D*np.log((t1 + D)/(2*U/Sc))
    return -Sc**2/(2*U) * (t1 - t2)

#generate dimensioned parameters
def generate_params_hyd1d(hg, lg, tg, p, ne, Sc, gam, hi, lam, sigma, rho, ai, theta, phi):

    Lh = lam*lg
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    b = b_fun(hg, gam, hi)
    ksat = ksat_fun(p, hg, lg, hi)
    E0 = E0_fun(theta, hg, tg)
    ds = ds_fun(hg, ne, gam, sigma, hi)
    tr = tr_fun(hg, p, ne, gam, sigma, hi, rho)
    tb = tb_fun(hg, p, ne, gam, sigma, hi, rho)
    pet = pet_fun(p, rho, ai)
    na = phi*ne

    return D, U, hg, lg, tg, E0, Lh, Sc, ksat, p, pet, b, ds, tr, tb, ne, na, gam, hi, lam, sigma, rho, ai, theta, phi

# params for both hyd1d recharge estimation and lem
sigma_all = np.array([8, 16, 32, 64, 128])
rho_all = np.array([0.05, 0.1, 0.2, 0.4, 0.8])

ai = 0.25
sc = 0.5
theta = 0.0
hi = 5.0
gam = 4.0
phi = 1.5
hg = 2.25
lg = 15 # geomorphic length scale [m]
tg = 22500*(365*24*3600) # geomorphic timescale [s]
ne = 0.1 # drainable porosity [-]
p = 1.0/(365*24*3600) # average precip rate

# for recharge estimation
lam = 10
bin_capacity_nd = 0.005 # bin capacity as a proportion of mean storm depth
Nt = 2000; Ny = 3; Nx = 50 # num timesteps, num y nodex, num x nodes

params = []
for sigma, rho in product(sigma_all, rho_all):
    params.append(generate_params_hyd1d(hg, lg, tg, p, ne, sc, gam, hi, lam, sigma, rho, ai, theta, phi))

df_params_1d = pd.DataFrame(np.array(params),columns=['D', 'U', 'hg', 'lg', 'tg', 'E0', 'Lh', 'Sc', 'ksat', 'p', 'pet', 'b', 'ds', 'tr', 'tb', 'ne', 'na', 'gam', 'hi', 'lam', 'sigma', 'rho', 'ai', 'theta', 'phi'])
df_params_1d['alpha'] = df_params_1d['hg']/df_params_1d['lg']
df_params_1d['Nx'] = Nx; df_params_1d['Ny'] = Ny; df_params_1d['Nt'] = Nt
df_params_1d['Nz'] = round((df_params_1d['b']*df_params_1d['na'])/(bin_capacity_nd*df_params_1d['ds']))
df_params_1d.loc[ID].to_csv('df_params_1d_%d.csv'%ID, index=True)

### recharge estimation

# parameters
ks = df_params_1d['ksat'][ID]
pet = df_params_1d['pet'][ID]
ne = df_params_1d['ne'][ID]
na = df_params_1d['na'][ID]
b = df_params_1d['b'][ID]
ds = df_params_1d['ds'][ID]
tr = df_params_1d['tr'][ID]
tb = df_params_1d['tb'][ID]
Lh = df_params_1d['Lh'][ID]
D = df_params_1d['D'][ID]
U = df_params_1d['U'][ID]
sc = df_params_1d['Sc'][ID]
Nz = df_params_1d['Nz'][ID]

# initialize grid
grid = RasterModelGrid((Ny, Nx), xy_spacing=Lh/Nx)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
x = grid.x_of_node
z =  calc_z(x, sc, U, D) - calc_z(x[-1], sc, U, D)
z = np.fliplr(z.reshape(grid.shape))
elev[:] = z.flatten()
base = grid.add_zeros('node', 'aquifer_base__elevation')
base[:] = elev - b
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev

# initialize landlab and DupuitLEM components
gdp = GroundwaterDupuitPercolator(grid,
                                  porosity=ne,
                                  hydraulic_conductivity=ks,
                                  recharge_rate=0.0,
                                  vn_coefficient=0.5,
                                  courant_coefficient=0.5,
                                  )
pdr = PrecipitationDistribution(grid,
                               mean_storm_duration=tr,
                               mean_interstorm_duration=tb,
                               mean_storm_depth=ds,
                               total_t=Nt*(tr+tb))
pdr.seed_generator(seedval=1235)
svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                available_water_content=na,
                 profile_depth=b,
                 num_bins=Nz,
                 )
svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
hm = HydrologyEventVadoseStreamPower(
                                    grid,
                                    precip_generator=pdr,
                                    groundwater_model=gdp,
                                    vadose_model=svm,
                                    )
#print("components initialized")

# run once to spin up model
hm.run_step()
#print("Spinup completed")

# run and record state
hm.run_step_record_state()
#print("Run record state finished")

#### Params for lem
RE = hm.cum_recharge / hm.cum_precip # recharge efficiency

v0_nd = 2.0 # contour width (also grid spacing)
Nx_lem = 125

Tg_nd = 2000 # total duration in units of tg [-]
dtg_nd = 2e-3 # maximum geomorphic timestep in units of tg [-]
ksf_base = 500 # morphologic scaling factor
Th_nd = 5 # hydrologic time in units of vn timescale [-]
output_interval = 1000

fields = ['D', 'U', 'hg', 'lg', 'tg', 'E0', 'Lh', 'Sc', 'ksat', 'p', 'b', 'ne', 'na', 'gam', 'hi', 'lam', 'sigma', 'rho', 'ai', 'theta', 'tr', 'tb', 'ds', 'pet']
df_params = df_params_1d.loc[ID,fields]
df_params['RE'] = RE
df_params['v0'] = v0_nd*df_params['lg']
df_params['Nx'] = Nx_lem
df_params['K'] = K_fun(df_params.v0, df_params.lg, df_params.tg)
df_params['alpha'] = df_params['hg']/df_params['lg']
df_params['td'] = (df_params['lg']*df_params['ne'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]
df_params['hc'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['dtg'] = dtg_nd*df_params['tg'] # geomorphic timestep [s]
df_params['Th'] = Th_nd*(df_params['ne']*df_params['v0']**2)/(4*df_params['ksat']*df_params['b']) #von neumann cond time [s]
df_params['ksf'] = df_params['dtg']/df_params['Th'] # morphologic scaling factor
df_params['output_interval'] = output_interval

df_params.to_csv('parameters.csv', index=True)
