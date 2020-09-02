
"""
Use StochasticRechargeShearStress model to test different values of
hydraulic conductivity. Save output.

19 May 2020
"""
import os
import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    FlowDirectorD8,
    FlowAccumulator,
    LakeMapperBarnes,
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

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#parameters
Ks_all = np.array([0.01, 0.05, 0.1, 0.5, 1.0])*(1/3600) #[m/s]
Ks = Ks_all[ID]
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
T_h = 30*24*3600 # total hydrological time
T_m = 2.5e6*(365*24*3600) # total simulation time [s]

p_seed = 2 #s eed for stochastic precipitation
storm_dt = 2*3600 # storm duration [s]
interstorm_dt = 48*3600 # interstorm duration [s]
p_d = 0.01 # storm depth [m]

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

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,d_k) # hydraulic conductivity [m/s]
ss_erosion_fun = bind_erosion_from_shear_stress(tauc,k_st,b_st)
ss_chezy_fun = bind_shear_stress_chezy(c_chezy=chezy_c)

#initialize grid
np.random.seed(2)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
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
hm = HydrologyEventShearStress(
        grid,
        precip_generator=pd,
        groundwater_model=gdp,
        flow_director=fd,
        flow_accumulator=fa,
        lake_mapper=lmb,
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
