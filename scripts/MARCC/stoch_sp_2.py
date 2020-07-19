"""
Stochastic recharge + constant thickness + StreamPowerModel

Vary l (length scale), keep everything else constant.

Date: 24 Jun 2020
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
def K_sp_fun(l, n, Pe, gam, lam, pi):
    return (Pe*lam*pi)/(gam*l**2*n)

def D_fun(l, p, n, gam, lam, pi):
    return (lam*pi*p*l)/(gam*n)

def ksat_fun(p, gam, phi):
    return (p*phi)/gam

def U_fun(p, n, pi):
    return (p*pi)/n

def b_fun(l, gam, om):
    return (l*gam)/om

#generate dimensioned parameters
def generate_parameters(p, l, n, gam, pe, lam, pi, phi, om):

    b = b_fun(l, gam, om)
    ksat = ksat_fun(p, gam, phi)
    U = U_fun(p, n, pi)
    D = D_fun(l, p, n, gam, lam, pi)
    K = K_sp_fun(l, n, pe, gam, lam, pi)

    return K, D, U, ksat, p, b, l, n, gam, pe, lam, pi, phi, om

#parameters
MSF = 500 # morphologic scaling factor [-]
T_h = 30*24*3600 # total hydrological time
T_m = 5e6*(365*24*3600) # total simulation time [s]

pe1 = 500
phi1 = 100
lam1 = 0.1
pi1 = 5e-6
om1 = 20
p1 = 1/(365*24*3600) # recharge rate [m/s]
n1 = 0.1 # drainable porosity []
gam1 = 0.2
l1 = np.geomspace(20,500,11) #equilibrium depth [m]

tr = 2*3600 # storm duration [s]
tb = 48*3600 # interstorm duration [s]
ds = p1*(tr+tb) # storm depth [m]

params = np.zeros((len(l1),14))
for i in range(len(l1)):

    params[i,:] = generate_parameters(p1, l1[i], n1, gam1, pe1, lam1, pi1, phi1, om1)

df_params = pd.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'l', 'n', 'gam', 'pe', 'lam', 'pi', 'phi', 'om'])
df_params['tr'] = tr
df_params['tb'] = tb
df_params['ds'] = ds

pickle.dump(df_params, open('parameters.p','wb'))

Ksp = df_params['K'][ID] #streampower coefficient
D = df_params['D'][ID] #hillslope diffusivity
U = df_params['U'][ID] #uplift Rate
Ks = df_params['ksat'][ID]
K0 = Ks*0.01
p = df_params['p'][ID]
b = df_params['b'][ID]
n = df_params['n'][ID]


output = {}
output["output_interval"] = 1000
output["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        ]
output["base_output_path"] = './data/stoch_sp_2_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,b) # hydraulic conductivity [m/s]

#initialize grid
np.random.seed(1234)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = b + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid, porosity=n, hydraulic_conductivity=ksat_fun, \
                                  regularization_f=0.01, recharge_rate=0.0, \
                                  courant_coefficient=0.9, vn_coefficient = 0.9)
pd = PrecipitationDistribution(grid, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
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
rm = RegolithConstantThicknessPerturbed(grid, equilibrium_depth=b, uplift_rate=U, std=1e-2, seed=1236)

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
