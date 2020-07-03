"""
Stochastic recharge + constant thickness + StreamPowerModel

Vary Beta (drainage index) and Taui (intensity index), keep everything else constant.

Date: 4 Jun 2020
"""
import os
import numpy as np
from itertools import product
import pandas as pd
import pickle

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
    PrecipitationDistribution,
    )
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import HydrologyEventStreamPower, RegolithConstantThicknessPerturbed
from DupuitLEM.grid_functions.grid_funcs import (
    bind_avg_hydraulic_conductivity
    )

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#dim equations
def K_sp_fun(beq, n, Pe, gam, lam, pi, om):
    return (Pe*gam*lam*pi)/(beq**2*n*om**2)

def D_fun(beq, p, n, gam, lam, pi, om):
    return (beq*p*lam*pi*om)/(n*gam**2)

def ksat_fun(p, beta, gam, om):
    return (beta*p*om)/(gam**2)

def U_fun(p, n, pi):
    return (p*pi)/n

def l_fun(beq, gam, om):
    return (om*beq)/gam

def tr_fun(p, beq, n, beta, taui, taub):
    return taui*taub*(beq*n)/(beta*p)

def td_fun(p, beq, n, beta, taui, taub):
    return (taub - taui*taub)*(beq*n)/(beta*p)

#generate dimensioned parameters
def generate_parameters(p, beq, n, gam, pe, lam, pi, beta, om, taui, taub):

    l = l_fun(beq, gam, om)
    ksat = ksat_fun(p, beta, gam, om)
    U = U_fun(p, n, pi)
    D = D_fun(beq, p, n, gam, lam, pi, om)
    K = K_sp_fun(beq, n, pe, gam, lam, pi, om)
    tr = tr_fun(p, beq, n, beta, taui, taub)
    td = td_fun(p, beq, n, beta, taui, taub)

    return K, D, U, ksat, p, tr, td, beq, l, n, gam, pe, lam, pi, phi, om, taui, taub

#parameters
MSF = 500 # morphologic scaling factor [-]
T_h = 30*24*3600 # total hydrological time
T_m = 1e7*(365*24*3600) # total simulation time [s]

taui_all = np.geomspace(0.01,0.99,8)
beta_all = np.geomspace(0.1,1000,6)
taui1 = np.array(list(product(taui_all,beta_all)))[:,0]
beta1 = np.array(list(product(taui_all,beta_all)))[:,1]
pe1 = 500
phi1 = 50
lam1 = 0.1
pi1 = 5e-6
om1 = 20
p1 = 1/(365*24*3600) # recharge rate [m/s]
n1 = 0.1 # drainable porosity []
gam1 = 0.2
beq1 = 1 #equilibrium depth [m]

storm_dt = 2*3600 # storm duration [s]
interstorm_dt = 48*3600 # interstorm duration [s]
p_d = p1*(storm_dt+interstorm_dt) # storm depth [m]

params = np.zeros((len(pe1),18))
for i in range(len(pe1)):

    params[i,:] = generate_parameters(p1, beq1, n1, gam1, pe1[i], lam1, pi1, phi1[i], om1)

df_params = pd.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'tr', 'td', 'beq', 'l', 'n', 'gam', 'pe', 'lam', 'pi', 'phi', 'om', 'taui', 'taub'])
df_params['depth'] = p1*(df_params['tr']+df_params['td'])

pickle.dump(df_params, open('parameters.p','wb'))

Ksp = df_params['K'][ID] #streampower coefficient
D = df_params['D'][ID] #hillslope diffusivity
U = df_params['U'][ID] #uplift Rate
Ks = df_params['ksat'][ID]
K0 = Ks*0.01
p = df_params['p'][ID]
beq = df_params['beq'][ID]
n = df_params['n'][ID]
tr = df_params['tr'][ID]
td = df_params['td'][ID]
p_d = df_params['depth'][ID]


output = {}
output["output_interval"] = 2500
output["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        ]
output["base_output_path"] = './data/stoch_sp_3_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,beq) # hydraulic conductivity [m/s]

#initialize grid
np.random.seed(1234)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = beq + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid, porosity=n, hydraulic_conductivity=ksat_fun, \
                                  regularization_f=0.01, recharge_rate=0.0, \
                                  courant_coefficient=0.9, vn_coefficient = 0.9)
pd = PrecipitationDistribution(grid, mean_storm_duration=tr,
    mean_interstorm_duration=td, mean_storm_depth=p_d,
    total_t=T_h)
pd.seed_generator(seedval=1235)
ld = LinearDiffuser(grid, linear_diffusivity=D)

#initialize other models
hm = HydrologyEventStreamPower(
        grid,
        precip_generator=pd,
        groundwater_model=gdp,
)
#use surface_water_effective__discharge for stochastic case
sp = FastscapeEroder(grid, K_sp=Ksp, m_sp=1, n_sp=1, discharge_field="surface_water_effective__discharge")
rm = RegolithConstantThicknessPerturbed(grid, equilibrium_depth=beq, uplift_rate=U, std=1e-2, seed=1236)

mdl = StreamPowerModel(grid,
        hydrology_model=hm,
        diffusion_model=ld,
        streampower_model=sp,
        regolith_model=rm,
        morphologic_scaling_factor=MSF,
        total_morphological_time=T_m,
        verbose=True,
        output_dict=output,
)

mdl.run_model()
