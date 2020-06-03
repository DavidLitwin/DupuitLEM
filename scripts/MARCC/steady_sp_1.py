"""
steady recharge + constant thickness + StreamPowerModel

Date: 16 Mar 2020
"""
import os
import numpy as np

from landlab import imshow_grid
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
    )
from DupuitLEM import StreamPowerModel
from DupuitLEM.auxiliary_models import HydrologySteadyStreamPower, RegolithConstantThickness
from DupuitLEM.grid_functions.grid_funcs import bind_avg_hydraulic_conductivity

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#parameters
p = 1.75/(365*24*3600) # recharge rate [m/s]
Ks_all = np.array([0.01, 0.05, 0.1, 0.5, 1.0])*(1/3600) #[m/s]
Ks = Ks_all[0]
K0 = 0.01*Ks # asymptotic hydraulic conductivity at infinite depth
d_k = 1 #m
n = 0.1 # porosity []
r = 0.01 # regularization factor
c = 0.9 # courant_coefficient
vn = 0.9 # von Neumann coefficient

d_eq = 1 #equilibrium depth [m]
U = 1E-4/(365*24*3600) # uniform uplift [m/s]
b_st = 1.5 #shear stress erosion exponent
k_st = 1e-10 #shear stress erosion coefficient
sp_c = 0.0 #threshold streampower
chezy_c = 15 #chezy coefficient for flow depth calcualtion
rho = 1000 #density [kg/m3]
g = 9.81 #gravitational constant [m/s2]
dx = 10 #grid cell width [m]
Ksp = k_st*( (rho*g)/(chezy_c*dx)**(2/3) )**b_st
m = 2/3*b_st
n = 2/3*b_st
D = 0.001/(365*24*3600) # hillslope diffusivity [m2/s]

MSF = 500 # morphologic scaling factor [-]
dt_h = 1e5 # hydrological timestep
T_m = 1e5*(365*24*3600) # total simulation time [s]

output = {}
output["output_interval"] = 1000
output["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        "surface_water__discharge",
        "groundwater__specific_discharge_node",
        ]
output["base_output_path"] = './data/steady_sp_1_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,d_k) # hydraulic conductivity [m/s]

#initialize grid
np.random.seed(2)
grid = RasterModelGrid((20, 26), xy_spacing=dx)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = d_eq + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid, porosity=n, hydraulic_conductivity=ksat_fun, \
                                  regularization_f=r, recharge_rate=p, \
                                  courant_coefficient=c, vn_coefficient = vn)
ld = LinearDiffuser(grid, linear_diffusivity = D)

#initialize other models
hm = HydrologySteadyStreamPower(
        grid,
        groundwater_model=gdp,
        hydrological_timestep=dt_h,
)
#use surface_water__discharge for steady case
sp = FastscapeEroder(grid, K_sp = Ksp, m_sp = m, n_sp=n, discharge_field='surface_water__discharge')
rm = RegolithConstantThickness(grid, equilibrium_depth=d_eq, uplift_rate=U)

mdl = StreamPowerModel(grid,
        hydrology_model = hm,
        diffusion_model = ld,
        streampower_model = sp,
        regolith_model = rm,
        morphologic_scaling_factor = MSF,
        total_morphological_time = T_m,
        verbose=True,
        output_dict = output,
)


mdl.run_model()
