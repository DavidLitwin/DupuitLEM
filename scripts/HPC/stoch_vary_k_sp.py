
"""
Use StochasticRechargeStreamPower model to test different values of
hydraulic conductivity. Save output.

28 May 2020
"""

import numpy as np
import os

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    FlowDirectorD8,
    FlowAccumulator,
    LakeMapperBarnes,
    LinearDiffuser,
    FastscapeEroder,
    PrecipitationDistribution,
    )
from DupuitLEM import StochasticRechargeStreamPower
from DupuitLEM.auxiliary_models import HydrologyEventStreamPower, RegolithConstantThickness
from DupuitLEM.grid_functions.grid_funcs import (
    bind_avg_hydraulic_conductivity
    )

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#parameters
Ks_all = np.array([0.01, 0.05, 0.1, 0.5, 1.0])*(1/3600) #[m/s]
Ks = Ks_all[ID]
K0 = 0.01*Ks # asymptotic hydraulic conductivity at infinite depth
d_k = 1 #m
n = 0.1 # porosity []
r = 0.01 # regularization factor
c = 0.9 # courant_coefficient
vn = 0.9 # von Neumann coefficient

d_eq = 1 #equilibrium depth [m]
U = 1E-4/(365*24*3600) # uniform uplift [m/s]
b_st = 1.5 #shear stress erosion exponent
k_st = 5e-11 #shear stress erosion coefficient
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
T_h = 30*24*3600 # total hydrological time
T_m = 2.5e6*(365*24*3600) # total simulation time [s]

p_seed = 2 #s eed for stochastic precipitation
storm_dt = 2*3600 # storm duration [s]
interstorm_dt = 48*3600 # interstorm duration [s]
p_d = 0.01 # storm depth [m]

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,d_k) # hydraulic conductivity [m/s]

output = {}
output["output_interval"] = 1000
output["output_fields"] = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        ]
output["base_output_path"] = './data/stoch_vary_k_sp_'
output["run_id"] = ID #make this task_id if multiple runs

#initialize grid
np.random.seed(2)
grid = RasterModelGrid((100, 100), xy_spacing=dx)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED,
                                 top=grid.BC_NODE_IS_CLOSED,
                                 left=grid.BC_NODE_IS_FIXED_VALUE,
                                 bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
elev[:] = d_eq + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()

#initialize landlab components
gdp = GroundwaterDupuitPercolator(grid, porosity=n, hydraulic_conductivity=ksat_fun, \
                                  regularization_f=r, \
                                  courant_coefficient=c, vn_coefficient = vn)
fd = FlowDirectorD8(grid)
fa = FlowAccumulator(grid,
				        surface='topographic__elevation',
						flow_director=fd,
						runoff_rate='average_surface_water__specific_discharge')
lmb = LakeMapperBarnes(grid, method='D8', fill_flat=False,
						  surface='topographic__elevation',
						  fill_surface='topographic__elevation',
						  redirect_flow_steepest_descent=False,
						  reaccumulate_flow=False,
						  track_lakes=False,
						  ignore_overfill=True)
ld = LinearDiffuser(grid, linear_diffusivity = D)

pd = PrecipitationDistribution(grid, mean_storm_duration=storm_dt,
    mean_interstorm_duration=interstorm_dt, mean_storm_depth=p_d,
    total_t=T_h)
pd.seed_generator(seedval=p_seed)

#initialize other models
hm = HydrologyEventStreamPower(
        grid,
        precip_generator=pd,
        groundwater_model=gdp,
        flow_director=fd,
        flow_accumulator=fa,
        lake_mapper=lmb,
)

sp = FastscapeEroder(grid, K_sp = Ksp, m_sp = m, n_sp=n, discharge_field='surface_water_effective__discharge')
rm = RegolithConstantThickness(grid, equilibrium_depth=d_eq, uplift_rate=U)

mdl = StochasticRechargeStreamPower(grid,
        hydrology_model = hm,
        diffusion_model = ld,
        erosion_model = sp,
        regolith_model = rm,
        morphologic_scaling_factor = MSF,
        total_morphological_time = T_m,
        verbose=True,
        output_dict = output,
)

mdl.run_model()
