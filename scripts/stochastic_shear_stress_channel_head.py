
"""
A closer look at a 1st order drainage basin to investigate channel head dynamics
using the StochasticRechargeShearStress model.

Date: 29 April 2020
"""

import numpy as np

from landlab import RasterModelGrid, imshow_grid
from DupuitLEM import StochasticRechargeShearStress
from DupuitLEM.grid_functions.grid_funcs import bind_avg_hydraulic_conductivity


#parameters
params = {}
Ks = 0.5/3600 #[m/s]
K0 = 0.01*Ks # asymptotic hydraulic conductivity at infinite depth
d_k = 1 #m
params["hydraulic_conductivity"] = bind_avg_hydraulic_conductivity(Ks,K0,d_k) #this is the depth-dependent K form
params["porosity"] = 0.2 #[]
params["regularization_factor"] = 0.01
params["courant_coefficient"] = 0.5
params["vn_coefficient"] = 0.8

params["permeability_production_rate"] = 2E-4/(365*24*3600) #[m/s]
params["characteristic_w_depth"] = 1  #m
params["uplift_rate"] = 1E-4/(365*24*3600) # uniform uplift [m/s]
params["b_st"] = 1.5 #shear stress erosion exponent
params["k_st"] = 1e-10 #shear stress erosion coefficient
params["shear_stress_threshold"] = 0.0 #threshold shear stress [N/m2]
params["chezy_c"] = 15 #chezy coefficient for flow depth calcualtion
params["hillslope_diffusivity"] = 0.001/(365*24*3600) # hillslope diffusivity [m2/s]

params["morphologic_scaling_factor"] = 500 # morphologic scaling factor [-]
params["total_hydrological_time"] = 30*24*3600 # total hydrological time
params["total_morphological_time"] = 1e6*(365*24*3600) # total simulation time [s]

params["precipitation_seed"] = 2
params["mean_storm_duration"] = 2*3600
params["mean_interstorm_duration"] = 48*3600
params["mean_storm_depth"] = 0.01 #[m]

# additional fields for saving output as netCDFs
params["output_interval"] = 100
params["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        "surface_water__discharge",
        "groundwater__specific_discharge_node",
        ]
params["base_output_path"] = 'C:/Users/dgbli/Documents/Research_Data/Landscape evolution/stoch_channel_head_'
params["run_id"] = 2 #make this SLURM task_id if multiple runs



"""
From the looks of the final state after the simulation, you will need about 20x20
grid to get the whole drainage basin in the top left corner.

"""
# whole grid
np.random.seed(2)
grid = RasterModelGrid((100, 100), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
d_i = -params["characteristic_w_depth"]*np.log(params["uplift_rate"]/params["permeability_production_rate"])
elev[:] = d_i + 0.1*np.random.rand(len(elev))
# base = grid.add_zeros('node', 'aquifer_base__elevation')
# wt = grid.add_zeros('node', 'water_table__elevation')
# wt[:] = elev.copy()


# partial grid that you want
x_indices = np.where(grid.x_of_node <= 250)
y_indices = np.where(grid.y_of_node >= 800)
indices = np.intersect1d(x_indices,y_indices)

grid_1 = RasterModelGrid((20, 26), xy_spacing=10.0)
grid_1.set_status_at_node_on_edges(right=grid_1.BC_NODE_IS_CLOSED, top=grid_1.BC_NODE_IS_CLOSED, \
                              left=grid_1.BC_NODE_IS_FIXED_VALUE, bottom=grid_1.BC_NODE_IS_CLOSED)
elev_1 = grid_1.add_zeros('node', 'topographic__elevation')
d_i = -params["characteristic_w_depth"]*np.log(params["uplift_rate"]/params["permeability_production_rate"])
elev_1[:] = elev[indices]
base_1 = grid_1.add_zeros('node', 'aquifer_base__elevation')
wt_1 = grid_1.add_zeros('node', 'water_table__elevation')
wt_1[:] = elev_1.copy()

#add partial grid to dictionary
params["grid"] = grid_1

#initialize the model
mdl = StochasticRechargeShearStress(params,save_output=True,verbose=True)

#run the model in one go for the total morphological time
mdl.run_model()

"""
Alternate way to run the model:

for i in range(N):
    mdl.run_step(dt_m)

where dt_m is the length (in seconds) of the morphological timestep. Note that
you will still need to specify the hydrological timestep dt_h in the parameter
dictionary, but that morphologic_scaling_factor and total_morphological_time
are optional arguments. When the run_model method is used, dt_m = MSF*dt_h.
"""

"""
The simplest way to visualize output is landlab's built in imshow_grid function.
Handles reshaping, axes, etc.

#elevation
plt.figure(figsize=(8,6))
imshow_grid(grid_1,elev_1, cmap='gist_earth', colorbar_label = 'Elevation [m]', grid_units=('m','m'))

# regolith thickness
plt.figure(figsize=(8,6))
imshow_grid(grid_1,elev_1-base_1, cmap='Blues', limits=(0,6), colorbar_label = 'regolith thickness', grid_units=('m','m'))
"""
