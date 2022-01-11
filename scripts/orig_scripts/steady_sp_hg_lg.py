"""
Steady recharge + constant thickness + StreamPowerModel

This script uses characteristic scales and dimensionless parameters presented
in Litwin et al. 2021. Here we test rescaling the domain with lg and hg,
observing the dependence of the solution on these length scales under different
conditions.

Date: 1 Sept 2020
"""
import os
import numpy as np
import pandas
import pickle
from itertools import product

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
    )
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import (
    HydrologySteadyStreamPower,
    RegolithConstantThickness,
    )

#slurm info
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

#generate dimensioned parameters
def generate_parameters(p, n, v0, hg, lg, tg, gam, hi):

    K = K_fun(v0, lg, tg)
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    b = b_fun(hg, gam, hi)
    ksat = ksat_fun(p, hg, lg, hi)

    return K, D, U, ksat, p, b, n, v0, hg, lg, tg, gam, hi

#parameters
hi1 = 5.0
gam1 = 2.5
lg_1 = np.array([15, 30, 60]) # geomorphic length scale [m]
hg_1 = np.array([2.25, 4.5, 9]) # geomorphic height scale [m]
lg_all = np.array(list(product(lg_1, hg_1)))[:,0]
hg_all = np.array(list(product(lg_1, hg_1)))[:,1]
tg = 22500*(365*24*3600) # geomorphic timescale [s]
n1 = 0.1 # drainable porosity [-]
p1 = 0.75/(365*24*3600) # steady recharge rate
v0_all = 0.7*lg_all # contour width (also grid spacing) [m]

Tg_nd = 1000 # total duration in units of tg [-]
dtg_nd = 2e-3 # geomorphic timestep in units of tg [-]
Th_nd = 5 # hydrologic time in units of t_vn [-]

# assemble parameters dataframe
params = np.zeros((len(lg_all),13))
for i in range(len(lg_all)):
    params[i,:] = generate_parameters(p1, n1, v0_all[i], hg_all[i], lg_all[i], tg, gam1, hi1)
df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'n', 'v0', 'hg', 'lg', 'tg', 'gam', 'hi'])
df_params['alpha'] = df_params['hg']/df_params['lg']
df_params['td'] = (df_params['lg']*df_params['n'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]
df_params['hc'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['dtg'] = dtg_nd*df_params['tg'] # geomorphic timestep [s]
df_params['Th'] = Th_nd*(df_params['n']*df_params['v0']**2)/(4*df_params['ksat']*df_params['b']) #von neumann cond time [s]
df_params['ksf'] = df_params['dtg']/df_params['Th'] # morphologic scaling factor
pickle.dump(df_params, open('parameters.p','wb'))

# pull values for this run
ksat = df_params['ksat'][ID]
p = df_params['p'][ID]
b = df_params['b'][ID]
n = df_params['n'][ID]

K = df_params['K'][ID]
v0 = df_params['v0'][ID]
Ksp = K/p # recharge rate from Q* goes in K
D = df_params['D'][ID]
U = df_params['U'][ID]
hg = df_params['hg'][ID]
lg = df_params['lg'][ID]

Th = df_params['Th'][ID]
Tg = df_params['Tg'][ID]
ksf = df_params['ksf'][ID]

output = {}
output["output_interval"] = 1000
output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = './data/steady_sp_5_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid
np.random.seed(12345)
grid = RasterModelGrid((125, 125), xy_spacing=v0)
grid.set_status_at_node_on_edges(
        right=grid.BC_NODE_IS_CLOSED,
        top=grid.BC_NODE_IS_CLOSED,
        left=grid.BC_NODE_IS_FIXED_VALUE,
        bottom=grid.BC_NODE_IS_CLOSED,
)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = b + 0.1*hg*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()

#initialize components
gdp = GroundwaterDupuitPercolator(grid,
        porosity=n,
        hydraulic_conductivity=ksat,
        regularization_f=0.01,
        recharge_rate=p,
        courant_coefficient=0.1,
        vn_coefficient = 0.1,
)
ld = LinearDiffuser(grid, linear_diffusivity=D)

hm = HydrologySteadyStreamPower(
        grid,
        groundwater_model=gdp,
        hydrological_timestep=Th,
)

# surface_water_area_norm__discharge (Q/sqrt(A)) = Q* p v0 sqrt(a)
sp = FastscapeEroder(grid,
        K_sp=Ksp,
        m_sp=1,
        n_sp=1,
        discharge_field="surface_water_area_norm__discharge",
)
rm = RegolithConstantThickness(grid, equilibrium_depth=b, uplift_rate=U)

mdl = StreamPowerModel(grid,
        hydrology_model=hm,
        diffusion_model=ld,
        erosion_model=sp,
        regolith_model=rm,
        morphologic_scaling_factor=ksf,
        total_morphological_time=Tg,
        verbose=True,
        output_dict=output,
)

mdl.run_model()
