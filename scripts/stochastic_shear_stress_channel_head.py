
"""
A closer look at a 1st order drainage basin to investigate channel head dynamics
using the StochasticRechargeShearStress model.

Date: 29 April 2020
"""
import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    PrecipitationDistribution,
    )
from DupuitLEM import StochasticRechargeShearStress
from DupuitLEM.auxiliary_models import HydrologyEventShearStress, RegolithConstantThickness
from DupuitLEM.grid_functions.grid_funcs import (
    bind_avg_hydraulic_conductivity,
    bind_erosion_from_shear_stress,
    bind_shear_stress_chezy,
    )

#parameters
Ks_all = np.array([0.01, 0.05, 0.1, 0.5, 1.0])*(1/3600) #[m/s]
Ks = Ks_all[0]
K0 = 0.01*Ks # asymptotic hydraulic conductivity at infinite depth
d_k = 1 #m
n = 0.2 # porosity []
r = 0.01 # regularization factor
c = 0.9 # courant_coefficient
vn = 0.9 # von Neumann coefficient

d_eq = 1 #equilibrium depth [m]
U = 1E-4/(365*24*3600) # uniform uplift [m/s]
b_st = 1.5 #shear stress erosion exponent
k_st = 1e-10 #shear stress erosion coefficient
tauc = 0.0 #threshold shear stress [N/m2]
chezy_c = 15 #chezy coefficient for flow depth calcualtion
D = 0.001/(365*24*3600) # hillslope diffusivity [m2/s]

MSF = 100 # morphologic scaling factor [-]
T_h = 30*24*3600 # total hydrological time
T_m = 1e5*(365*24*3600) # total simulation time [s]

p_seed = 2 #s eed for stochastic precipitation
storm_dt = 2*3600 # storm duration [s]
interstorm_dt = 48*3600 # interstorm duration [s]
p_d = 0.01 # storm depth [m]

output = {}
output["output_interval"] = 500
output["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        "surface_water__discharge",
        "groundwater__specific_discharge_node",
        ]
output["base_output_path"] = 'C:/Users/dgbli/Documents/Research_Data/Landscape evolution/stoch_channel_head_'
output["run_id"] = 5 #make this task_id if multiple runs

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,d_k) # hydraulic conductivity [m/s]
ss_erosion_fun = bind_erosion_from_shear_stress(tauc,k_st,b_st)
ss_chezy_fun = bind_shear_stress_chezy(c_chezy=chezy_c)

#initialize grid
# whole grid
np.random.seed(2)
grid_1 = RasterModelGrid((100, 100), xy_spacing=10.0)
grid_1.set_status_at_node_on_edges(right=grid_1.BC_NODE_IS_CLOSED, top=grid_1.BC_NODE_IS_CLOSED, \
                              left=grid_1.BC_NODE_IS_FIXED_VALUE, bottom=grid_1.BC_NODE_IS_CLOSED)
elev_1 = grid_1.add_zeros('node', 'topographic__elevation')
elev_1[:] = d_eq + 0.1*np.random.rand(len(elev_1))

# partial grid that you want
x_indices = np.where(grid_1.x_of_node <= 250)
y_indices = np.where(grid_1.y_of_node >= 800)
indices = np.intersect1d(x_indices,y_indices)

grid = RasterModelGrid((20, 26), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = elev_1[indices]
base = grid.add_zeros('node', 'aquifer_base__elevation')
base[:] = elev-d_eq
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()


#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid, porosity=n, hydraulic_conductivity=ksat_fun, \
                                  regularization_f=r, \
                                  courant_coefficient=c, vn_coefficient = vn)
ld = LinearDiffuser(grid, linear_diffusivity = D)
pd = PrecipitationDistribution(grid, mean_storm_duration=storm_dt,
    mean_interstorm_duration=interstorm_dt, mean_storm_depth=p_d,
    total_t=T_h)
pd.seed_generator(seedval=p_seed)

#initialize other models
hm = HydrologyEventShearStress(
        grid,
        precip_generator=pd,
        groundwater_model=gdp,
        shear_stress_function=ss_chezy_fun,
        erosion_rate_function=ss_erosion_fun,
)

rm = RegolithConstantThickness(grid, equilibrium_depth=d_eq, uplift_rate=U)

mdl = StochasticRechargeShearStress(grid,
        hydrology_model = hm,
        diffusion_model = ld,
        regolith_model = rm,
        morphologic_scaling_factor = MSF,
        total_morphological_time = T_m,
        verbose=True,
        output_dict = output,
)

mdl.run_model()

