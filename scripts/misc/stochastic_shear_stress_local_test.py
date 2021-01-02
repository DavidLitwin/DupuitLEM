"""
test of stochastic recharge + constant thickness + ShearStressModel, without saving output

19 May 2020
"""

import numpy as np

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    PrecipitationDistribution,
    )
from DupuitLEM import ShearStressModel
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

MSF = 500 # morphologic scaling factor [-]
T_h = 30*24*3600 # total hydrological time
T_m = 1e4*(365*24*3600) # total simulation time [s]

p_seed = 2 #s eed for stochastic precipitation
storm_dt = 2*3600 # storm duration [s]
interstorm_dt = 48*3600 # interstorm duration [s]
p_d = 0.01 # storm depth [m]

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

mdl = ShearStressModel(grid,
        hydrology_model = hm,
        diffusion_model = ld,
        regolith_model = rm,
        morphologic_scaling_factor = MSF,
        total_morphological_time = T_m,
        verbose=True,
)

#%% Run whole model

mdl.run_model()

#%% run hydrological model

for i in range(100):

    hm.run_step()

    print('finished step ' + str(i))

#%% run whole model one step at a time

for i in range(100):

    mdl.run_step(mdl.dt_m)

    print('finished step ' + str(i))

#%% run and track storms and substeps

max_substeps_storm = np.zeros(1000)
max_substeps_interstorm = np.zeros(1000)

intensity = []
storm_dts = []
interstorm_dts = []

for i in range(1000):

    mdl.run_step(mdl.dt_m)

    max_substeps_storm[i] = mdl.max_substeps_storm
    max_substeps_interstorm[i] = mdl.max_substeps_interstorm

    print('Completed model loop %d' % i)

    intensity.append(mdl.intensities)
    storm_dts.append(mdl.storm_dts)
    interstorm_dts.append(mdl.interstorm_dts)

#%% Run groundwater model only

max_substeps_storm = np.zeros(500)
max_substeps_interstorm = np.zeros(500)

for j in range(500):

    hm.generate_exp_precip()

    num_substeps_storm = np.zeros(len(hm.storm_dts))
    num_substeps_interstorm = np.zeros(len(hm.storm_dts))

    for i in range(len(hm.storm_dts)):

        #run event, accumulate flow, and calculate resulting shear stress
        gdp.recharge_rate = hm.intensities[i]
        gdp.run_with_adaptive_time_step_solver(hm.storm_dts[i])
        num_substeps_storm[i] = gdp.number_of_substeps


        #run interevent, accumulate flow, and calculate resulting shear stress
        gdp.recharge_rate = 0.0
        gdp.run_with_adaptive_time_step_solver(hm.interstorm_dts[i])
        num_substeps_interstorm[i] = gdp.number_of_substeps

    max_substeps_storm[j] = max(num_substeps_storm)
    max_substeps_interstorm[j] = max(num_substeps_interstorm)

    print('completed '+str(j))
