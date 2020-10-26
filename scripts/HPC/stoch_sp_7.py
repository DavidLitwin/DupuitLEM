"""
Stochastic recharge + constant thickness + StreamPowerModel

This script uses dimensionless parameters based on Theodoratos method of
nondimensionalizing the governing landscape evolution equation. Corresponding
with the phase space diagram, vary beta and gamma for a few different values
of rho.

\[lambda] == (ks U^2)/(p K D),
\[Gamma] == (ks b U)/(p Dm),
\[Alpha] == (i tr)/(b n),
\[Rho] == p/i = tr/(tb + tr)

Date: 14 Sept 2020
"""
import os
import numpy as np
from itertools import product
import pandas
import pickle

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
    PrecipitationDistribution,
    )
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import (
    HydrologyEventStreamPower,
    RegolithConstantThickness,
    )

#slurm info
task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#dim equations
def b_fun(U, K, gam, lam):
    return (U*gam)/(K*lam)

def ksat_fun(D, U, K, p, lam):
    return (D*K*p*lam)/(U**2)

def ds_fun(U, K, n, beta, lam):
    return (U*n*beta)/(K*lam)

def tr_fun(U, K, p, n, beta, lam, rho):
    return (U*n*beta*rho)/(K*p*lam)

def tb_fun(U, K, p, n, beta, lam, rho):
    return (U*n*beta)*(1-rho)/(K*p*lam)


#generate dimensioned parameters
def generate_parameters(D, U, K, p, n, beta, gam, lam, rho):

    b = b_fun(U, K, gam, lam)
    ksat = ksat_fun(D, U, K, p, lam)
    ds = ds_fun(U, K, n, beta, lam)
    tr = tr_fun(U, K, p, n, beta, lam, rho)
    tb = tb_fun(U, K, p, n, beta, lam, rho)

    return K, D, U, ksat, p, b, ds, tr, tb, n, beta, gam, lam, rho

#parameters
beta_all = np.array([0.01, 0.1, 1.0, 2.0])
gam_all = np.array([0.5, 1.0, 2.0, 4.0])
rho_all = np.array([0.01, 0.1, 0.5])
beta1 = np.array(list(product(beta_all, gam_all, rho_all)))[:,0]
gam1 = np.array(list(product(beta_all, gam_all, rho_all)))[:,1]
rho1 = np.array(list(product(beta_all, gam_all, rho_all)))[:,2]
lam1 = 0.5
lg = 15 # geomorphic length scale
D1 = 0.01/(365*24*3600) # hillslope linear diffusivity [m2/s]
U1 = 1e-4/(365*24*3600) # Uplift rate [m/s]
K1 = (D1/lg**2) # Streampower incision coefficient [1/s]
p1 = 0.75/(365*24*3600) # average rainfall rate [m/s]
n1 = 0.1 # drainable porosity [-]

Tg_nd = 3000 # total duration in units of tg [-]
dtg_max_nd = 2e-3 # maximum geomorphic timestep in units of tg [-]
MSF = 5000 # morphologic scaling factor
Th_nd = 20 # hydrologic time in units of (tr+tb) [-]
dx_nd = 1.2 # dimensionless grid spacing [-]

params = np.zeros((len(beta1),14))
for i in range(len(beta1)):

    params[i,:] = generate_parameters(D1, U1, K1, p1, n1, beta1[i], gam1[i], lam1, rho1[i])

df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'ds', 'tr', 'tb', 'n', 'beta', 'gam', 'lam', 'rho'])
df_params['alpha'] = df_params['beta']/df_params['gam'] # dimensionless number alpha
df_params['hg'] = df_params['U']/df_params['K'] # characteristic geomorphic vertical length scale [m]
df_params['lg'] = np.sqrt(df_params['D']/df_params['K']) # characteristic geomorphic horizontal length scale [m]
df_params['tg'] = 1/df_params['K'] # characteristic geomorphic timescale [s]
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb']) # hydrologic simulation time [s]
df_params['ibar'] = df_params['p']/df_params['rho'] # mean storm rainfall intensity [m/s]
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['MSF'] = MSF
df_params['dtg'] = df_params['MSF']*df_params['Th'] # geomorphic timestep [s]
df_params['dtg_max'] = dtg_max_nd*df_params['tg'] # the maximum duration of a geomorphic substep [s]
df_params['dx_nd'] = dx_nd

pickle.dump(df_params, open('parameters.p','wb'))

ksat = df_params['ksat'][ID]
p = df_params['p'][ID]
b = df_params['b'][ID]
n = df_params['n'][ID]
tr = df_params['tr'][ID]
tb = df_params['tb'][ID]
ds = df_params['ds'][ID]

K = df_params['K'][ID]
Ksp = K/p #see governing equation. If the discharge field is (Q/sqrt(A)) then streampower coeff is K/p
D = df_params['D'][ID]
U = df_params['U'][ID]
hg = df_params['hg'][ID]
lg = df_params['lg'][ID]

Th = df_params['Th'][ID]
Tg = df_params['Tg'][ID]
MSF = df_params['MSF'][ID]
dtg_max = df_params['dtg_max'][ID]

output = {}
output["output_interval"] = (10/(df_params['dtg']/df_params['tg'])).round().astype(int)[ID] #save output every 10 tg
output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = './data/stoch_sp_7_'
output["run_id"] = ID #make this task_id if multiple runs

# postrun_ss_cond = {}
# postrun_ss_cond['stop_at_rate'] = 1e-3*U
# postrun_ss_cond['how'] = 'percentile'
# postrun_ss_cond['percentile_value'] = 90

#initialize grid
np.random.seed(12345)
grid = RasterModelGrid((125, 125), xy_spacing=dx_nd*lg)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = b + 0.1*hg*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid, porosity=n, hydraulic_conductivity=ksat, \
                                  regularization_f=0.01, recharge_rate=0.0, \
                                  courant_coefficient=0.9, vn_coefficient = 0.9)
pd = PrecipitationDistribution(grid, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=Th)
pd.seed_generator(seedval=1235)
ld = LinearDiffuser(grid, linear_diffusivity=D)

#initialize other models
hm = HydrologyEventStreamPower(
        grid,
        precip_generator=pd,
        groundwater_model=gdp,
)

#use surface_water_area_norm__discharge (Q/sqrt(A)) for Theodoratos definitions
sp = FastscapeEroder(grid, K_sp=Ksp, m_sp=1, n_sp=1, discharge_field="surface_water_area_norm__discharge")
rm = RegolithConstantThickness(grid, equilibrium_depth=b, uplift_rate=U)

mdl = StreamPowerModel(grid,
        hydrology_model=hm,
        diffusion_model=ld,
        erosion_model=sp,
        regolith_model=rm,
        morphologic_scaling_factor=MSF,
        maximum_morphological_dt=dtg_max,
        total_morphological_time=Tg,
        verbose=True,
        output_dict=output,
        # steady_state_condition=postrun_ss_cond,
)

mdl.run_model()
