"""
test of steady recharge + constant thickness + ShearStress model, without saving output

19 May 2020
"""

import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FlowDirectorD8,
    FlowAccumulator,
    LakeMapperBarnes,
    )
from DupuitLEM import ShearStressModel
from DupuitLEM.auxiliary_models import HydrologySteadyShearStress, RegolithConstantThickness
from DupuitLEM.grid_functions.grid_funcs import (
    bind_avg_hydraulic_conductivity,
    bind_erosion_from_shear_stress,
    bind_shear_stress_chezy,
    )

#parameters
R = 1.5/(365*24*3600) # recharge rate [m/s]
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

#initialize other models
hm = HydrologySteadyShearStress(
        grid,
        groundwater_model=gdp,
        flow_director=fd,
        flow_accumulator=fa,
        lake_mapper=lmb,
        shear_stress_function=ss_chezy_fun,
        erosion_rate_function=ss_erosion_fun,
        hydrological_timestep = dt_h
)

rm = RegolithConstantThickness(grid, equilibrium_depth=d_eq, uplift_rate=U)

mdl = ShearStressModel(grid,
        hydrology_model = hm,
        diffusion_model = ld,
        regolith_model = rm,
        morphologic_scaling_factor = MSF,
        total_morphological_time = T_m,
        verbose=True,
)

#%%

mdl.run_model()

#%%

for i in range(100):

    hm.run_step()

    print('finished step ' + str(i))

#%%

for i in range(100):

    mdl.run_step(mdl.dt_m)

    print('finished step ' + str(i))
