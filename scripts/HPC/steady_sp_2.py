"""
Steady recharge + constant thickness + StreamPowerModel

This script uses dimensionless parameters based on Theodoratos method of
nondimensionalizing the governing landscape evolution equation. Vary eta
and gamma.

\[Eta] == (b Km)/U,
\[Gamma] == (ks b U)/(i Dm),

Date: 18 Aug 2020
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
def b_fun(U, K, eta):
    return (U*eta)/K

def ksat_fun(D, U, p, K, gam, eta):
    return (D*p*K*gam)/(U**2*eta)

#generate dimensioned parameters
def generate_parameters(D, U, K, p, n, gam, eta):

    b = b_fun(U, K, eta)
    ksat = ksat_fun(D, U, p, K, gam, eta)

    return K, D, U, ksat, p, b, n, gam, eta

#parameters
eta_all = np.geomspace(0.1,10,5)
gam_all = np.geomspace(0.01,1.0,5)
D1 = 0.01/(365*24*3600) # hillslope linear diffusivity [m2/s]
U1 = 1e-4/(365*24*3600) # Uplift rate [m/s]
K1 = 1e-4/(365*24*3600) # Streampower incision coefficient [1/s]
n1 = 0.1 # drainable porosity [-]
p1 = 1/(365*24*3600) # steady precipitation rate

Tg_nd = 800 # total duration in units of tg [-]
dtg_nd = 5e-3 # geomorphic timestep in units of tg [-]
Th_nd = 5 # hydrologic time in units of t_vn [-]

eta1 = np.array(list(product(eta_all, gam_all)))[:,0]
gam1 = np.array(list(product(eta_all, gam_all)))[:,1]

params = np.zeros((len(eta1),9))
for i in range(len(eta1)):

    params[i,:] = generate_parameters(D1, U1, K1, p1, n1, gam1[i], eta1[i])

df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'n', 'gam', 'eta'])
df_params['hg'] = df_params['U']/df_params['K'] # characteristic geomorphic vertical length scale [m]
df_params['lg'] = np.sqrt(df_params['D']/df_params['K']) # characteristic geomorphic horizontal length scale [m]
df_params['tg'] = 1/df_params['K'] # characteristic geomorphic timescale [s]
df_params['tfill'] = (df_params['n']*df_params['b'])/df_params['p']
df_params['tdrain'] = (df_params['D']*df_params['n'])/(df_params['ksat']*df_params['U'])
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
output["base_output_path"] = './data/steady_sp_2_'
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
ld = LinearDiffuser(grid, linear_diffusivity=D)

#initialize other models
hm = HydrologySteadyStreamPower(
        grid,
        groundwater_model=gdp,
        hydrological_timestep=Th,
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
