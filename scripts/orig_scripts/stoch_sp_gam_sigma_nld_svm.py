"""
Stochastic recharge + constant thickness + StreamPowerModel

This script uses characteristic scales and dimensionless parameters presented
in Litwin et al. 2021, but extends using parameters Sigma and Rho that
describe storage vs storm depth and rainfall steadiness respectively.


Date: 7 Oct 2021
"""
import os
import numpy as np
from itertools import product
import pandas
import pickle

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    TaylorNonLinearDiffuser,
    FastscapeEroder,
    PrecipitationDistribution,
    )
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import (
    HydrologyEventVadoseStreamPower,
    RegolithConstantThickness,
    SchenkVadoseModel,
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

def ds_fun(hg, n, gam, sigma, hi):
    return (hg*n*gam)/(hi*sigma)

def tr_fun(hg, p, n, gam, sigma, hi, rho):
    return (hg*n*gam*rho)/(p*sigma*hi)

def tb_fun(hg, p, n, gam, sigma, hi, rho):
    return (hg*n*gam)*(1-rho)/(p*sigma*hi)

#generate dimensioned parameters
def generate_parameters(p, n, v0, lg, tg, alpha, gam, hi, sigma, rho, ai):

    hg = alpha*lg
    K = K_fun(v0, lg, tg)
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    b = b_fun(hg, gam, hi)
    ksat = ksat_fun(p, hg, lg, hi)
    ds = ds_fun(hg, n, gam, sigma, hi)
    tr = tr_fun(hg, p, n, gam, sigma, hi, rho)
    tb = tb_fun(hg, p, n, gam, sigma, hi, rho)
    pet = ai*p

    return K, D, U, ksat, p, pet, b, n, v0, hg, lg, tg, ds, tr, tb, alpha, gam, hi, sigma, rho, ai


#parameters
sigma_all = np.array([8, 16, 32, 64, 128])
gam_all = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

ai = 0.5
sc = 0.5
hi = 5.0
rho = 0.03
alpha = 0.15
lg = 15 # geomorphic length scale [m]
tg = 22500*(365*24*3600) # geomorphic timescale [s]
v0 = 2.0*lg # contour width (also grid spacing) [m]
n = 0.1 # drainable porosity [-]
p = 1.0/(365*24*3600) # steady recharge rate
Srange = 0.2 # range of relative saturation
Nz = 500 # number of bins in vadose model

Tg_nd = 2000 # total duration in units of tg [-]
dtg_max_nd = 2e-3 # maximum geomorphic timestep in units of tg [-]
ksf_base = 500 # morphologic scaling factor
Th_nd = 50 # hydrologic time in units of (tr+tb) [-]

params = []
for sigma, gam in product(sigma_all, gam_all):
    params.append(generate_parameters(p, n, v0, lg, tg, alpha, gam, hi, sigma, rho, ai))

df_params = pandas.DataFrame(np.array(params),columns=['K', 'D', 'U', 'ksat', 'p', 'pet', 'b', 'n', 'v0', 'hg', 'lg', 'tg', 'ds', 'tr', 'tb', 'alpha', 'gam', 'hi', 'sigma', 'rho', 'ai'])
df_params['Sc'] = sc
df_params['td'] = (df_params['lg']*df_params['n'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]
df_params['beta'] = (df_params['tr']+df_params['tb'])/df_params['td']
df_params['ha'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['Srange'] = Srange
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['ksf'] = ksf_base/df_params['beta'] # morphologic scaling factor
df_params['Th'] = Th_nd*(df_params['tr']+df_params['tb']) # hydrologic simulation time [s]
df_params['dtg'] = df_params['ksf']*df_params['Th'] # geomorphic timestep [s]
df_params['dtg_max'] = dtg_max_nd*df_params['tg'] # the maximum duration of a geomorphic substep [s]
df_params['output_interval'] = (10/(df_params['dtg']/df_params['tg'])).round().astype(int)

pickle.dump(df_params, open('parameters.p','wb'))

ksat = df_params['ksat'][ID]
p = df_params['p'][ID]
pet = df_params['pet'][ID]
Srange = df_params['Srange'][ID]
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
Sc = df_params['Sc'][ID]

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
output["base_output_path"] = './data/stoch_sp_gam_sigma_svm_'
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
pdr = PrecipitationDistribution(grid, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=Th)
pdr.seed_generator(seedval=1235)
nld = TaylorNonLinearDiffuser(grid, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True)


#initialize other models
svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                 available_relative_saturation=Srange,
                 profile_depth=b,
                 porosity=n,
                 num_bins=Nz,
                 )
hm = HydrologyEventVadoseStreamPower(
                                    grid,
                                    precip_generator=pdr,
                                    groundwater_model=gdp,
                                    vadose_model=svm,
                                    )

#use surface_water_area_norm__discharge (Q/sqrt(A)) for Theodoratos definitions
sp = FastscapeEroder(grid, K_sp=Ksp, m_sp=1, n_sp=1, discharge_field="surface_water_area_norm__discharge")
rm = RegolithConstantThickness(grid, equilibrium_depth=b, uplift_rate=U)

mdl = StreamPowerModel(grid,
        hydrology_model=hm,
        diffusion_model=nld,
        erosion_model=sp,
        regolith_model=rm,
        morphologic_scaling_factor=ksf,
        maximum_morphological_dt=dtg_max,
        total_morphological_time=Tg,
        verbose=False,
        output_dict=output,
)

mdl.run_model()
