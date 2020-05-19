
"""
Use StochasticRechargeShearStress model to test different values of
hydraulic conductivity. Save output.

Date: 3 April 2020
"""
import os
import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    )
from DupuitLEM import SteadyRechargeShearStress
from DupuitLEM.runners import HydrologySteadyShearStress, RegolithConstantThickness
from DupuitLEM.grid_functions.grid_funcs import (
    bind_avg_hydraulic_conductivity,
    bind_erosion_from_shear_stress,
    bind_shear_stress_chezy,
    )

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#parameters
R = 1.752/(365*24*3600) #[m/s] #equivalent to 2 hr storm dt, 48 hr interstorm dt, 0.01m depth.
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
k_st = 5e-11 #shear stress erosion coefficient
tauc = 0.0 #threshold shear stress [N/m2]
chezy_c = 15 #chezy coefficient for flow depth calcualtion
D = 0.001/(365*24*3600) # hillslope diffusivity [m2/s]

MSF = 500 # morphologic scaling factor [-]
dt_h = 1e5 # total hydrological time
T_m = 2.5e6*(365*24*3600) # total simulation time [s]

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,d_k) # hydraulic conductivity [m/s]
ss_erosion_fun = bind_erosion_from_shear_stress(tauc,k_st,b_st)
ss_chezy_fun = bind_shear_stress_chezy(c_chezy=chezy_c)

output = {}
output["output_interval"] = 1000
output["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        "surface_water__discharge",
        "groundwater__specific_discharge_node",
        ]
output["base_output_path"] = './data/stoch_vary_k_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid
np.random.seed(2)
grid = RasterModelGrid((20, 25), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = d_eq + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid, porosity=n, hydraulic_conductivity=ksat_fun, \
                                  regularization_f=r, recharge_rate = R, \
                                  courant_coefficient=c, vn_coefficient = vn)
ld = LinearDiffuser(grid, linear_diffusivity = D)

#initialize other models
hm = HydrologySteadyShearStress(
        grid,
        groundwater_model=gdp,
        shear_stress_function=ss_chezy_fun,
        erosion_rate_function=ss_erosion_fun,
)

rm = RegolithConstantThickness(grid, equilibrium_depth=d_eq, uplift_rate=U)

mdl = SteadyRechargeShearStress(grid,
        hydrology_model = hm,
        diffusion_model = ld,
        regolith_model = rm,
        hydrological_timestep = dt_h,
        morphologic_scaling_factor = MSF,
        total_morphological_time = T_m,
        verbose=True,
)

mdl.run_model()
