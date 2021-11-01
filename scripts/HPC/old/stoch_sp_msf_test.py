"""
Stochastic recharge + constant thickness + StreamPowerModel

This script uses characteristic scales and dimensionless parameters presented
in Litwin et al. 2021, but extends using parameters Beta and Rho that
describe rainfall steadiness and storage variability respectively.

Test the effect of the morphoplogic scaling factor on the results for a
single combination of dimensionless parameters.

Date: 2 July 2021
"""
import os
import numpy as np
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

def ds_fun(hg, n, beta, hi):
    return (hg*n*beta)/hi

def tr_fun(hg, p, n, beta, hi, rho):
    return (hg*n*beta*rho)/(p*hi)

def tb_fun(hg, p, n, beta, hi, rho):
    return (hg*n*beta)*(1-rho)/(p*hi)

#generate dimensioned parameters
def generate_parameters(p, n, v0, hg, lg, tg, gam, hi, beta, rho):

    K = K_fun(v0, lg, tg)
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    b = b_fun(hg, gam, hi)
    ksat = ksat_fun(p, hg, lg, hi)
    ds = ds_fun(hg, n, beta, hi)
    tr = tr_fun(hg, p, n, beta, hi, rho)
    tb = tb_fun(hg, p, n, beta, hi, rho)

    return K, D, U, ksat, p, b, n, v0, hg, lg, tg, ds, tr, tb, gam, hi, beta, rho


#parameters
hi = 5.0
beta = 0.1
gam = 2.5
rho = 0.01
lg = 15 # geomorphic length scale [m]
hg = 2.25 # geomorphic height scale [m]
tg = 22500*(365*24*3600) # geomorphic timescale [s]
v0 = 1.2*lg # contour width (also grid spacing) [m]
n = 0.1 # drainable porosity [-]
p = 0.75/(365*24*3600) # steady recharge rate

Tg_nd = 1500 # total duration in units of tg [-]
dtg_max_nd = 2e-3 # maximum geomorphic timestep in units of tg [-]
ksf_all = np.array([1250, 2500, 5000, 10000, 20000, 40000]) # morphologic scaling factor
Th_nd = 20 # hydrologic time in units of (tr+tb) [-]

params = np.zeros((len(ksf_all),18))
for i in range(len(ksf_all)):
    params[i,:] = generate_parameters(p, n, v0, hg, lg, tg, gam, hi, beta, rho)

df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'n', 'v0', 'hg', 'lg', 'tg', 'ds', 'tr', 'tb', 'gam', 'hi', 'beta', 'rho'])
df_params['alpha'] = df_params['hg']/df_params['lg']
df_params['td'] = (df_params['lg']*df_params['n'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]
df_params['hc'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['ksf'] = ksf_all  # morphologic scaling factor
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb']) # hydrologic simulation time [s]
df_params['dtg'] = df_params['ksf']*df_params['Th'] # geomorphic timestep [s]
df_params['dtg_max'] = dtg_max_nd*df_params['tg'] # the maximum duration of a geomorphic substep [s]
df_params['output_interval'] = (10/(df_params['dtg']/df_params['tg'])).round().astype(int)
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
ksf = df_params['ksf'][ID]
dtg_max = df_params['dtg_max'][ID]

output = {}
output["output_interval"] = df_params['output_interval'][ID] #save output every 10 tg
output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = './data/stoch_sp_gam_beta_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid
np.random.seed(12345)
grid = RasterModelGrid((125, 125), xy_spacing=v0)
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
        morphologic_scaling_factor=ksf,
        maximum_morphological_dt=dtg_max,
        total_morphological_time=Tg,
        verbose=False,
        output_dict=output,
)

mdl.run_model()