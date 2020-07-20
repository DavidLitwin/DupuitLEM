"""
Stochastic recharge + constant thickness + StreamPowerModel

This script uses dimensionless parameters based on Theodoratos method of
nondimensionalizing the governing landscape evolution equation.

\[Eta] == (b Km)/U,
\[Gamma] == (ksat b U)/(p Dm),
\[Alpha] == ds/(b n),
\[Tau]r == (tr ksat U)/(Dm n)

Date: 15 Jul 2020
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
    RegolithConstantThicknessPerturbed,
    RegolithConstantThickness,
    )
from DupuitLEM.grid_functions.grid_funcs import (
    bind_avg_hydraulic_conductivity
    )

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#dim equations
def K_sp_fun(U, b, eta):
    return (U*eta)/b

def ksat_fun(D, p, U, b, gam):
    return (D*p*gam)/(b*U)

def ds_fun(b, n, alpha):
    return b*n*alpha

def tr_fun(p, b, n, gam, taur):
    return (b*n*taur)/(p*gam)

def td_fun(p, b, n, alpha, gam, taur):
    return (b*n*alpha)/p - (b*n*taur)/(p*gam)

#generate dimensioned parameters
def generate_parameters(D, p, U, b, n, alpha, gam, eta, taur):

    K = K_sp_fun(U, b, eta)
    ksat = ksat_fun(D, p, U, b, gam)
    tr = tr_fun(p, b, n, gam, taur)
    td = td_fun(p, b, n, alpha, gam, taur)
    ds = ds_fun(b, n, alpha)

    return K, D, U, ksat, p, b, ds, tr, td, n, alpha, gam, eta, taur

#parameters
MSF = 500 # morphologic scaling factor [-]
T_h = 30*24*3600 # total hydrological time [s]
T_m = 5e6*(365*24*3600) # total simulation time [s]

### fix below here
D1 = 0.002/(365*24*3600) # hillslope linear diffusivity [m2/s]
p1 = 1/(365*24*3600) # recharge rate [m/s]
U1 = 1e-4/(365*24*3600) # Uplift rate [m/s]
b1 = 1.0 # permeable thickness [m]
n1 = 0.1 # drainable porosity []
eta_all = np.geomspace(0.1,100,6)
gam_all = np.geomspace(1,100,6)
eta1 = np.array(list(product(eta_all,gam_all)))[:,0]
gam1 = np.array(list(product(eta_all,gam_all)))[:,1]
alpha1 = 0.05
taur1 = 0.04

params = np.zeros((len(eta1),14))
for i in range(len(eta1)):

    params[i,:] = generate_parameters(D1, p1, U1, b1, n1, alpha1, gam1[i], eta1[i], taur1)

df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'ds', 'tr', 'tb', 'n', 'alpha', 'gam', 'eta', 'taur'])

pickle.dump(df_params, open('parameters.p','wb'))

K = df_params['K'][ID] #streampower coefficient
D = df_params['D'][ID] #hillslope diffusivity
U = df_params['U'][ID] #uplift Rate
Ks = df_params['ksat'][ID]
K0 = Ks*0.01
p = df_params['p'][ID]
b = df_params['b'][ID]
n = df_params['n'][ID]
tr = df_params['tr'][ID]
tb = df_params['tb'][ID]
ds = df_params['ds'][ID]

Ksp = K/p #see governing equation. If the discharge field is (Q/sqrt(A)) then streampower coeff is K/p

output = {}
output["output_interval"] = 1000
output["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        ]
output["base_output_path"] = './data/stoch_sp_3_'
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
sp = FastscapeEroder(grid, K_sp=Ksp, m_sp=1, n_sp=1, discharge_field="surface_water_area_norm__discharge")
rm = RegolithConstantThicknessPerturbed(grid, equilibrium_depth=b, uplift_rate=U, std=1e-2, seed=1236)
# rm = RegolithConstantThickness(grid, equilibrium_depth=b, uplift_rate=U)

mdl = StreamPowerModel(grid,
        hydrology_model=hm,
        diffusion_model=ld,
        erosion_model=sp,
        regolith_model=rm,
        morphologic_scaling_factor=MSF,
        total_morphological_time=T_m,
        verbose=True,
        output_dict=output,
)

mdl.run_model()
