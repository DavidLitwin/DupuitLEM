"""
Steady recharge + exponential production + StreamPowerModel

Very simple test

Date: 11 April 2025
"""
#%%
import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
    DepthDependentDiffuser,
    )
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import (
    HydrologySteadyStreamPower,
    RegolithExponentialProduction,
    RegolithConstantThickness,
    )

#%%
ID = 0

# pull values for this run
ksat = 1e-7
p = 0.5/(365*24*3600)
n = 0.05

K = 5e-5 / (365*24*3600)
Ksp = K/p # recharge rate from Q* goes in K
D = 0.01 / (365*24*3600)
U = 1e-4 / (365*24*3600)
C = 2 * U
b = 1
bstar = - b / np.log(U/C)

Th = 1e4
Tg = 6e5 * (365*24*3600)
ksf = 5e5
v0 = 10

def calc_gamma(U, D, p, ks, b):
    return (b*ks*U)/(p*D)

def calc_beta(K, U, D, v0, p, ks):
    return (ks*U**2)/(p*v0**(2/3)*D**(2/3)*K**(4/3))

def calc_alpha(K, U, D, v0):
    return  U/(v0**(1/3)*D**(1/3)*K**(2/3))

print(f'alpha: {calc_alpha(K, U, D, v0)}')
print(f'beta: {calc_beta(K, U, D, v0, p, ksat)}')
print(f'gamma: {calc_gamma(U,D,p,ksat,b)}')

# output = {}
# output["output_interval"] = 1000
# output["output_fields"] = [
#         "at_node:topographic__elevation",
#         "at_node:aquifer_base__elevation",
#         "at_node:water_table__elevation",
#         ]
# output["base_output_path"] = './data/steady_sp_gam_hi_'
# output["run_id"] = ID #make this task_id if multiple runs

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
elev[:] = b + 0.1*np.random.rand(len(elev))
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

b = grid.add_zeros('node', 'topographic__elevation')
# ld = DepthDependentDiffuser(grid, linear_diffusivity=D, soil_transport_decay_depth=bstar)


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

rm = RegolithExponentialProduction(grid, characteristic_depth=b, regolith_production_rate=2*U, uplift_rate=U)
# rm = RegolithConstantThickness(grid, equilibrium_depth=5, uplift_rate=U)

mdl = StreamPowerModel(grid,
        hydrology_model=hm,
        diffusion_model=ld,
        erosion_model=sp,
        regolith_model=rm,
        morphologic_scaling_factor=ksf,
        total_morphological_time=Tg,
        verbose=False,
        # output_dict=output,
)

#%%

mdl.run_model()

# %%
