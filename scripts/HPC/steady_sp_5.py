"""
Steady recharge + constant thickness + StreamPowerModel

This script uses dimensionless parameters based on modified Theodoratos/Bonetti
method of nondimensionalizing the governing landscape evolution equation.
The purpose here is to determine whether the scaling with lg remains when the
length scale and domain size are changed. Keep dimensionless parameters the same.

\[lambda] == (ks (hg/lg)^2)/p,
\[Gamma] == (ks b hg/lg)/(p lg),

Date: 1 Sept 2020
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
def K_fun(a0, lg, tg):
    return np.sqrt(lg)/(np.sqrt(a0)*tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

def b_fun(hg, gam, lam):
    return (hg*gam)/lam

def ksat_fun(p, hg, lg, lam):
    return (lg**2*p*lam)/hg**2

#generate dimensioned parameters
def generate_parameters(p, n, a0, hg, lg, tg, gam, lam):

    K = K_fun(a0, lg, tg)
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    b = b_fun(hg, gam, lam)
    ksat = ksat_fun(p, hg, lg, lam)

    return K, D, U, ksat, p, b, n, a0, hg, lg, tg, gam, lam

#parameters
lam1 = 5.0
gam1 = 2.5
lg_all = np.array([15, 30, 60]) # geomorphic length scale [m]
hg = 2.25 # geomorphic height scale [m]
tg = 22500*(365*24*3600) # geomorphic timescale [s]
a0 = 0.7*15 #valley width factor [m]
n1 = 0.1 # drainable porosity [-]
p1 = 0.75/(365*24*3600) # steady precipitation rate

Tg_nd = 3000 # total duration in units of tg [-]
dtg_nd = 2e-3 # geomorphic timestep in units of tg [-]
Th_nd = 5 # hydrologic time in units of t_vn [-]

params = np.zeros((len(lg_all),13))
for i in range(len(lg_all)):

    params[i,:] = generate_parameters(p1, n1, a0, hg, lg_all[i], tg, gam1, lam1)

df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'n', 'a0', 'hg', 'lg', 'tg', 'gam', 'lam'])
df_params['tfill'] = (df_params['n']*df_params['b'])/df_params['p']
df_params['tdrain'] = (df_params['lg']*df_params['n'])/(df_params['ksat']*df_params['hg']/df_params['lg'])
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['dtg'] = dtg_nd*df_params['tg'] # geomorphic timestep [s]
df_params['Th'] = Th_nd*(df_params['n']*0.8*df_params['lg'])/(4*df_params['ksat']*df_params['b']) # hydrologic simulation time in units of von neumann stability time [s]
df_params['MSF'] = df_params['dtg']/df_params['Th'] # morphologic scaling factor

pickle.dump(df_params, open('parameters.p','wb'))

ksat = df_params['ksat'][ID]
p = df_params['p'][ID]
b = df_params['b'][ID]
n = df_params['n'][ID]

K = df_params['K'][ID]
a0 = df_params['a0'][ID]
v0 = 0.7*df_params['lg'][ID] #min contour width (grid spacing) [m]
Ksp = K*np.sqrt(a0/v0)/p # see implementation section of paper
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

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid,
        porosity=n,
        hydraulic_conductivity=ksat,
        regularization_f=0.01,
        recharge_rate=p,
        courant_coefficient=0.1,
        vn_coefficient = 0.1,
)
ld = LinearDiffuser(grid, linear_diffusivity=D)

#initialize other models
hm = HydrologySteadyStreamPower(
        grid,
        groundwater_model=gdp,
        hydrological_timestep=Th,
)

#use surface_water_area_norm__discharge (Q/sqrt(A)) for Theodoratos definitions
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
        morphologic_scaling_factor=MSF,
        total_morphological_time=Tg,
        verbose=True,
        output_dict=output,
        # steady_state_condition=postrun_ss_cond,
)

mdl.run_model()
