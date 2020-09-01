"""
Stochastic recharge + constant thickness + StreamPowerModel

This script uses dimensionless parameters based on Theodoratos method of
nondimensionalizing the governing landscape evolution equation. Vary alpha
and rho for several different gamma-lambda combinations.

\[lambda] == (p K D)/(ks U^2),
\[Gamma] == (ks b U)/(p Dm),
\[Alpha] == (i tr)/(b n),
\[Rho] == p/i = tr/(tb + tr)

Date: 1 Sept 2020
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
    return (U*gam*lam)/K

def ksat_fun(D, U, tr, n, alpha, gam, rho):
    return (D*n*alpha*gam*rho)/(U*tr)

def ds_fun(U, K, n, alpha, gam, lam):
    return (U*n*alpha*gam*lam)/K

def tb_fun(tr, rho):
    return tr*(1-rho)/rho

def p_fun(U, K, tr, n, alpha, gam, lam, rho):
    return (U*n*alpha*gam*lam*rho)/(K*tr)

#generate dimensioned parameters
def generate_parameters(D, U, K, tr, n, alpha, gam, lam, rho):

    b = b_fun(U, K, gam, lam)
    ksat = ksat_fun(D, U, tr, n, alpha, gam, rho)
    tb = tb_fun(tr, rho)
    ds = ds_fun(U, K, n, alpha, gam, lam)
    p = p_fun(U, K, tr, n, alpha, gam, lam, rho)

    return K, D, U, ksat, p, b, ds, tr, tb, n, alpha, gam, lam, rho

#parameters
lam_all = np.geomspace(0.2,2.0,5)
gam_all = np.geomspace(0.5,5.0,5)
eta_all_prod = np.array(list(product(lam_all, gam_all)))[:,0]
gam_all_prod = np.array(list(product(lam_all, gam_all)))[:,1]
eta_gam_id = np.array([0, 4, 12, 20, 24])

alpha_all = np.array([0.01, 0.05, 0.1])
rho_all = np.array([0.01, 0.1, 0.5])

alpha1 = np.array(list(product(alpha_all, rho_all, eta_gam_id)))[:,0]
rho1 = np.array(list(product(alpha_all, rho_all, eta_gam_id)))[:,1]
eta_gam_id1 = np.array(list(product(alpha_all, rho_all, eta_gam_id)), dtype=int)[:,2]
eta1 = eta_all_prod[eta_gam_id1]
gam1 = gam_all_prod[eta_gam_id1]

D1 = 0.01/(365*24*3600) # hillslope linear diffusivity [m2/s]
U1 = 1e-4/(365*24*3600) # Uplift rate [m/s]
K1 = 1e-4/(365*24*3600) # Streampower incision coefficient [1/s]
tr1 = 4*3600 # mean storm duration [s]
n1 = 0.1 # drainable porosity [-]

Tg_nd = 800 # total duration in units of tg [-]
dtg_nd = 1e-2 # geomorphic timestep in units of tg [-]
Th_nd = 50 # hydrologic time in units of (tr+tb) [-]

params = np.zeros((len(eta1),14))
for i in range(len(eta1)):

    params[i,:] = generate_parameters(D1, U1, K1, tr1, n1, alpha1[i], gam1[i], eta1[i], rho1[i])

df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'ds', 'tr', 'tb', 'n', 'alpha', 'gam', 'eta', 'rho'])
df_params['hg'] = df_params['U']/df_params['K'] # characteristic geomorphic vertical length scale [m]
df_params['lg'] = np.sqrt(df_params['D']/df_params['K']) # characteristic geomorphic horizontal length scale [m]
df_params['tg'] = 1/df_params['K'] # characteristic geomorphic timescale [s]
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['dtg'] = dtg_nd*df_params['tg'] # geomorphic timestep [s]
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb']) # hydrologic simulation time [s]
df_params['ibar'] = df_params['p']/df_params['rho'] # mean storm rainfall intensity [m/s]
df_params['MSF'] = df_params['dtg']/df_params['Th'] # morphologic scaling factor

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

output = {}
output["output_interval"] = 1000
output["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        ]
output["base_output_path"] = './data/stoch_sp_4_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid
np.random.seed(12345)
grid = RasterModelGrid((125, 125), xy_spacing=0.8*lg)
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
        total_morphological_time=Tg,
        verbose=True,
        output_dict=output,
)

mdl.run_model()
