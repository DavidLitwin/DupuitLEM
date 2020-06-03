
"""
test of StochasticRechargeStreamPower model, without saving output

28 May 2020
"""

import numpy as np
from landlab import imshow_grid
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    LinearDiffuser,
    FastscapeEroder,
    PrecipitationDistribution,
    )
from DupuitLEM import StochasticRechargeStreamPower
from DupuitLEM.auxiliary_models import HydrologyEventStreamPower, RegolithConstantThickness
from DupuitLEM.grid_functions.grid_funcs import (
    bind_avg_hydraulic_conductivity
    )


#parameters
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
T_h = 30*24*3600 # total hydrological time
T_m = 1e5*(365*24*3600) # total simulation time [s]

p_seed = 2 #s eed for stochastic precipitation
storm_dt = 2*3600 # storm duration [s]
interstorm_dt = 22*3600 # interstorm duration [s]
p_d = 0.002 # storm depth [m]

#initialize grid_functions
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,d_k) # hydraulic conductivity [m/s]

#initialize grid
np.random.seed(2)
grid = RasterModelGrid((100, 100), xy_spacing=dx)
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
hm = HydrologyEventStreamPower(
        grid,
        precip_generator=pd,
        groundwater_model=gdp,
)

sp = FastscapeEroder(grid, K_sp = Ksp, m_sp = m, n_sp=n, discharge_field='surface_water_effective__discharge')
rm = RegolithConstantThickness(grid, equilibrium_depth=d_eq, uplift_rate=U)

mdl = StochasticRechargeStreamPower(grid,
        hydrology_model = hm,
        diffusion_model = ld,
        streampower_model = sp,
        regolith_model = rm,
        morphologic_scaling_factor = MSF,
        total_morphological_time = T_m,
        verbose=True,
)

#%% Run whole model

mdl.run_model()

#%% run hydrological model

q_eff = []
# intensities = []

for i in range(100):

    hm.run_step()
    
    q_eff.append(grid.at_node['surface_water_effective__discharge'])
    # intensities.append(hm.intensities)

    print('finished step ' + str(i))
    
q_effmax = np.zeros(len(q_eff))
for i in range(len(q_eff)):
    q_effmax[i] = np.max(q_eff[i])

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

for j in range(10):

    hm.generate_exp_precip()

    num_substeps_storm = np.zeros(len(hm.storm_dts))
    num_substeps_interstorm = np.zeros(len(hm.storm_dts))

    q_total_vol = np.zeros_like(hm.q_eff)
    q2 = np.zeros_like(hm.q_eff)
    for i in range(len(hm.storm_dts)):
        q0 = q2.copy() #save prev end of interstorm flow rate

        # print(hm.intensities[i])
        # print(hm.storm_dts[i])
        #run event, accumulate flow, and calculate resulting shear stress
        gdp.recharge_rate = hm.intensities[i]
        gdp.run_with_adaptive_time_step_solver(hm.storm_dts[i])
        _,q1 = hm.fa.accumulate_flow(update_flow_director=False)
        print(max(q1))
        # num_substeps_storm[i] = gdp.number_of_substeps

        
        #run interevent, accumulate flow, and calculate resulting shear stress
        gdp.recharge_rate = 0.0
        gdp.run_with_adaptive_time_step_solver(max(hm.interstorm_dts[i],1e-15))
        _,q2 = hm.fa.accumulate_flow(update_flow_director=False)
        print(max(q0))
        # num_substeps_interstorm[i] = gdp.number_of_substeps


        q_total_vol += 0.5*(q0+q1)*hm.storm_dts[i]
        # print(q_total_vol[0])
    
    q_eff = q_total_vol/hm.T_h
    print(max(q_eff))
    
    # print(np.isnan(q_eff).any())

    print('completed '+str(j))


#%%


hm.generate_exp_precip()

num_substeps_storm = np.zeros(len(hm.storm_dts))
num_substeps_interstorm = np.zeros(len(hm.storm_dts))

for i in range(len(hm.storm_dts)):

    #run event, accumulate flow, and calculate resulting shear stress
    gdp.recharge_rate = hm.intensities[i]
    gdp.run_with_adaptive_time_step_solver(hm.storm_dts[i])
    _,q1 = hm.fa.accumulate_flow(update_flow_director=False)
    # num_substeps_storm[i] = gdp.number_of_substeps


    #run interevent, accumulate flow, and calculate resulting shear stress
    gdp.recharge_rate = 0.0
    gdp.run_with_adaptive_time_step_solver(hm.interstorm_dts[i])
    _,q2 = hm.fa.accumulate_flow(update_flow_director=False)
    # num_substeps_interstorm[i] = gdp.number_of_substeps
    
    plt.figure()
    imshow_grid(grid,q2)