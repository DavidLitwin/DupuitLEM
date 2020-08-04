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
    RegolithConstantThickness,
    )

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#dim equations
def b_fun(U, K, eta):
    return (U*eta)/K

def ksat_fun(D, p, U, K, eta, gam, rho):
    return (D*p*K*gam*rho)/(U**2*eta)

def ds_fun(U, K, n, alpha, eta):
    return (U*n*alpha*eta)/K

def tr_fun(p, U, K, n, alpha, eta, rho):
    return (U*n*alpha*eta)/(p*K*rho)

def tb_fun(p, U, K, n, alpha, eta, rho):
    return U*n*alpha*eta*(1-rho)/(p*K*rho**2)

#generate dimensioned parameters
def generate_parameters(D, p, U, K, n, alpha, gam, eta, rho):

    b = b_fun(U, K, eta)
    ksat = ksat_fun(D, p, U, K, eta, gam, rho)
    tr = tr_fun(p, U, K, n, alpha, eta, rho)
    tb = tb_fun(p, U, K, n, alpha, eta, rho)
    ds = ds_fun(U, K, n, alpha, eta)

    return K, D, U, ksat, p, b, ds, tr, tb, n, alpha, gam, eta, rho

#parameters
MSF = 500 # morphologic scaling factor [-]
T_h = 30*24*3600 # total hydrological time [s]
T_m = 5e6*(365*24*3600) # total simulation time [s]

eta_all = np.geomspace(0.2,2,5)
gam_all = np.geomspace(0.001,0.1,5)
alpha1 = 0.01
rho1 = 0.005
D1 = 0.005/(365*24*3600) # hillslope linear diffusivity [m2/s]
U1 = 5e-5/(365*24*3600) # Uplift rate [m/s]
K1 = 1e-5/(365*24*3600) # Streampower incision coefficient [1/s]
p1 = 1/(365*24*3600) # long term average rainfall rate [m/s]
n1 = 0.1 # drainable porosity []

eta1 = np.array(list(product(eta_all, gam_all)))[:,0]
gam1 = np.array(list(product(eta_all, gam_all)))[:,1]

params = np.zeros((len(eta1),14))
for i in range(len(eta1)):

    params[i,:] = generate_parameters(D1, p1, U1, K1, n1, alpha1, gam1[i], eta1[i], rho1)

df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'ds', 'tr', 'tb', 'n', 'alpha', 'gam', 'eta', 'rho'])
df_params['hg'] = df_params['U']/df_params['K']
df_params['lg'] = np.sqrt(df_params['D']/df_params['K'])
df_params['tg'] = 1/df_params['K']
df_params['ibar'] = df_params['p']/df_params['rho']

pickle.dump(df_params, open('parameters.p','wb'))

K = df_params['K'][ID] #streampower coefficient
D = df_params['D'][ID] #hillslope diffusivity
U = df_params['U'][ID] #uplift Rate
ksat = df_params['ksat'][ID]
p = df_params['p'][ID]
b = df_params['b'][ID]
n = df_params['n'][ID]
tr = df_params['tr'][ID]
tb = df_params['tb'][ID]
ds = df_params['ds'][ID]
hg = df_params['hg'][ID]

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

#initialize grid
np.random.seed(1234)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
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
# rm = RegolithConstantThicknessPerturbed(grid, equilibrium_depth=b, uplift_rate=U, std=1e-2, seed=1236)
rm = RegolithConstantThickness(grid, equilibrium_depth=b, uplift_rate=U)

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
