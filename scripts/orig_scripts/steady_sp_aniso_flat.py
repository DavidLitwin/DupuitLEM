"""
Steady recharge + constant thickness + StreamPowerModel

This is a test of anisotropic hydraulic conductivity (in the x- and y-directions), otherwise
using the characteristic scales and dimensionless parameters presented in Litwin et al. 2021. 

Date: 14 August 2023
"""

#%%

import os
import numpy as np
import pandas
from numpy.linalg import inv

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
    )
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import (
    HydrologySteadyStreamPower,
    RegolithConstantBaselevel,
    )

def get_link_hydraulic_conductivity(grid, K):
    """Returns array of hydraulic conductivity on links, allowing for aquifers
    with laterally anisotropic hydraulic conductivity.

    Parameters
    ----------
    K: (2x2) array of floats (m/s)
        The hydraulic conductivity tensor:
        [[Kxx, Kxy],[Kyx,Kyy]]
    """

    u = grid.unit_vector_at_link
    K_link = np.zeros(len(u))
    for i in range(len(u)):
        K_link[i] = np.dot(np.dot(u[i, :], K), u[i, :])
    return K_link

#%%
#slurm info
task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
#%%

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
def generate_parameters(p, n, v0, lg, tg, alpha, gam, hi):

    hg = lg * alpha
    K = K_fun(v0, lg, tg)
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    b = b_fun(hg, gam, hi)
    ksat = ksat_fun(p, hg, lg, hi)

    return K, D, U, ksat, p, b, n, v0, hg, lg, tg, alpha, gam, hi

#parameters
hi = 5.0 # hillslope number
gam = 4.0 # drainage capacity index
lg = 15 # geomorphic length scale [m]
alpha = 0.5 # characteristic gradient
tg = 22500*(365*24*3600) # geomorphic timescale [s]
v0 = 1.5*lg # contour width (also grid spacing) [m]
n1 = 0.1 # drainable porosity [-]
p1 = 0.75/(365*24*3600) # steady recharge rate
rot = np.array([0, 30, 45, 60, 90])*np.pi/180 #rotation angle between grid axes and princpal axes of anisotropy
k_ratio = 50 # ratio of minimum to maximum princpal hydraulic conductivities

# time factors
Tg_nd = 200 # total duration in units of tg [-]
dtg_nd = 2e-3 # geomorphic timestep in units of tg [-]
Th = 30*24*3600 # hydrologic time per geomorphic step

# keep all parameters fixed (just repeat them)
params = np.array(generate_parameters(p1, n1, v0, lg, tg, alpha, gam, hi)).reshape(1,14)
params = np.repeat(params,len(rot), axis=0)
df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'ksat', 'p', 'b', 'n', 'v0', 'hg', 'lg', 'tg', 'alpha', 'gam', 'hi'])

# add a non-condant rotation
df_params['k_rot'] = rot
df_params['k_ratio'] = k_ratio
df_params['alpha'] = df_params['hg']/df_params['lg']
df_params['td'] = (df_params['lg']*df_params['n'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]
df_params['hc'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['Tg'] = Tg_nd*df_params['tg'] # Total geomorphic simulation time [s]
df_params['dtg'] = dtg_nd*df_params['tg'] # geomorphic timestep [s]
df_params['Th'] = Th
df_params['ksf'] = df_params['dtg']/df_params['Th'] # morphologic scaling factor

#%% specific run

df_params.loc[ID].to_csv('parameters.csv', index=True)

# pull values for this run
kmax = df_params['ksat'][ID]
kmin = kmax * 1/(df_params['k_ratio'][ID])
krot = df_params['k_rot'][ID]


p = df_params['p'][ID]
b = df_params['b'][ID]
n = df_params['n'][ID]

K = df_params['K'][ID]
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
output["base_output_path"] = './data/steady_aniso_'
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

# calculate link hydraulic conductivity
rot = np.array([[np.cos(krot), -np.sin(krot)],[np.sin(krot), np.cos(krot)]])
K_principal = np.array([[kmax,0.0],[0.0, kmin]])
K = rot @ K_principal @ inv(rot)
k1 = get_link_hydraulic_conductivity(grid, K)

#initialize components
gdp = GroundwaterDupuitPercolator(grid,
        porosity=n,
        hydraulic_conductivity=k1,
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
rm = RegolithConstantBaselevel(grid, uplift_rate=U)


#%%
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


# %%
