"""
Stochastic recharge + constant thickness + StreamPowerModel

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

def ksat_fun(p, gam, phi):
    return (p*phi)/gam

def U_fun(p, n, pi):
    return (p*pi)/n

def l_fun(beq, gam, om):
    return (om*beq)/gam

#generate dimensioned parameters
def generate_parameters(p, beq, n, gam, pe, lam, pi, phi, om):

    l = l_fun(beq, gam, om)
    ksat = ksat_fun(p, gam, phi)
    U = U_fun(p, n, pi)
    D = D_fun(beq, p, n, gam, lam, pi, om)
    K = K_sp_fun(beq, n, pe, gam, lam, pi, om)

    return K, D, U, ksat, p, beq, l, n

#parameters
MSF = 500 # morphologic scaling factor [-]
T_h = 30*24*3600 # total hydrological time
T_m = 2.5e6*(365*24*3600) # total simulation time [s]

pe_all = np.geomspace(100,1000,6)
lam1 = 0.1
pi1 = 5e-6
phi_all = np.geomspace(10,100,6)
om1 = 20
p1 = 1/(365*24*3600) # recharge rate [m/s]
n1 = 0.1 # drainable porosity []
gam1 = 0.2
beq1 = 1 #equilibrium depth [m]
pe1 = np.array(list(product(pe_all,phi_all)))[:,0]
phi1 = np.array(list(product(pe_all,phi_all)))[:,1]

storm_dt = 2*3600 # storm duration [s]
interstorm_dt = 48*3600 # interstorm duration [s]
p_d = p1*(storm_dt+interstorm_dt) # storm depth [m]

params = np.zeros((len(pe1),8))
for i in range(len(pe1)):

    params[i,:] = generate_parameters(p1, beq1, n1, gam1, pe1[i], lam1, pi1, phi1[i], om1)

df_params = pd.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'beq', 'l', 'n'])
df_params['storm_dt'] = storm_dt
df_params['interstorm_dt'] = interstorm_dt
df_params['depth'] = p_d

pickle.dump(df_params, open('parameters.p','wb'))

Ksp = df_params['K'][ID] #streampower coefficient
D = df_params['D'][ID] #hillslope diffusivity
U = df_params['U'][ID] #uplift Rate
Ks = df_params['ksat'][ID]
K0 = Ks*0.01
p = df_params['p'][ID]
beq = df_params['beq'][ID]
n = df_params['n'][ID]


output = {}
output["output_interval"] = 1000
output["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        "surface_water__discharge",
        "groundwater__specific_discharge_node",
        ]
output["base_output_path"] = './data/stoch_sp_1_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,beq) # hydraulic conductivity [m/s]

#initialize grid
np.random.seed(2)
grid = RasterModelGrid((50, 50), xy_spacing=10.0)
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
pd = PrecipitationDistribution(grid, mean_storm_duration=storm_dt,
    mean_interstorm_duration=interstorm_dt, mean_storm_depth=p_d,
    total_t=T_h)
pd.seed_generator(seedval=2)
ld = LinearDiffuser(grid, linear_diffusivity = D)

#initialize other models
hm = HydrologyEventStreamPower(
        grid,
        precip_generator=pd,
        groundwater_model=gdp,
)
#use surface_water__discharge for steady case
sp = FastscapeEroder(grid, K_sp = Ksp, m_sp = 1, n_sp=1, discharge_field='surface_water__discharge')
rm = RegolithConstantThicknessPerturbed(grid, equilibrium_depth=beq, uplift_rate=U)

mdl = StreamPowerModel(grid,
        hydrology_model = hm,
        diffusion_model = ld,
        streampower_model = sp,
        regolith_model = rm,
        morphologic_scaling_factor = MSF,
        total_morphological_time = T_m,
        verbose=True,
        output_dict = output,
)

mdl.run_model()
