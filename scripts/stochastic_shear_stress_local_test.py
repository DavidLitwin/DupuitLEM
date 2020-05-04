
"""
test of StochasticRechargeShearStress model, without saving output

Date: 3 April 2020
"""

import numpy as np

from landlab import RasterModelGrid
from DupuitLEM import StochasticRechargeShearStress
from DupuitLEM.grid_functions.grid_funcs import bind_avg_hydraulic_conductivity


#parameters
#parameters
params = {}
Ks_all = np.array([0.01, 0.05, 0.1, 0.5, 1.0])*(1/3600) #[m/s]
Ks = Ks_all[4]
K0 = 0.01*Ks # asymptotic hydraulic conductivity at infinite depth
d_k = 1 #m
params["hydraulic_conductivity"] = bind_avg_hydraulic_conductivity(Ks,K0,d_k)
params["porosity"] = 0.2 #[]
params["regularization_factor"] = 0.01
params["courant_coefficient"] = 0.5
params["vn_coefficient"] = 0.8

params["permeability_production_rate"] = 2E-4/(365*24*3600) #[m/s]
params["characteristic_w_depth"] = 1  #m
params["uplift_rate"] = 1E-4/(365*24*3600) # uniform uplift [m/s]
params["b_st"] = 1.5 #shear stress erosion exponent
params["k_st"] = 1e-10 #shear stress erosion coefficient
params["shear_stress_threshold"] = 0.01 #threshold shear stress [N/m2]
params["manning_n"] = 0.05 #manning's n for flow depth calcualtion
params["hillslope_diffusivity"] = 0.001/(365*24*3600) # hillslope diffusivity [m2/s]

params["morphologic_scaling_factor"] = 500 # morphologic scaling factor [-]
params["total_hydrological_time"] = 30*24*3600 # total hydrological time
params["total_morphological_time"] = 2.5e6*(365*24*3600) # total simulation time [s]

params["precipitation_seed"] = 2
params["mean_storm_duration"] = 2*3600
params["mean_interstorm_duration"] = 48*3600
params["mean_storm_depth"] = 0.01

#initialize grid
np.random.seed(2)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
d_i = -params["characteristic_w_depth"]*np.log(params["uplift_rate"]/params["permeability_production_rate"])
elev[:] = d_i + 0.1*np.random.rand(len(elev))
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev.copy()

params["grid"] = grid

mdl = StochasticRechargeShearStress(params,save_output=False,verbose=True)

mdl.run_model()

#%%
# from landlab.grid.mappers import map_max_of_node_links_to_node

# storm_vn = np.zeros(len(mdl.storm_dts), len(elev))
# interstorm_vn = np.zeros(len(mdl.storm_dts), len(elev))
# storm_courant = np.zeros(len(mdl.storm_dts), len(elev))
# interstorm_courant = np.zeros(len(mdl.storm_dts), len(elev))

num_substeps_storm = np.zeros(len(mdl.storm_dts))
num_substeps_interstorm = np.zeros(len(mdl.storm_dts))


for i in range(len(mdl.storm_dts)):

    #run event, accumulate flow, and calculate resulting shear stress
    mdl.gdp.recharge_rate = mdl.intensities[i]
    mdl.gdp.run_with_adaptive_time_step_solver(mdl.storm_dts[i])
    num_substeps_storm[i] = mdl.gdp.number_of_substeps


    # storm_vn[i,:] = map_max_of_node_links_to_node(grid, mdl.gdp._vn_criteria)
    # storm_courant[i,:] = map_max_of_node_links_to_node(grid, mdl.gdp._courant_criteria)

    #run interevent, accumulate flow, and calculate resulting shear stress
    mdl.gdp.recharge_rate = 0.0
    mdl.gdp.run_with_adaptive_time_step_solver(mdl.interstorm_dts[i])
    num_substeps_interstorm[i] = mdl.gdp.number_of_substeps



    # interstorm_vn[i,:] = map_max_of_node_links_to_node(grid, mdl.gdp._vn_criteria)
    # interstorm_courant[i,:] = map_max_of_node_links_to_node(grid, mdl.gdp._courant_criteria)

