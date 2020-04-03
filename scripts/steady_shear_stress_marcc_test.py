
"""
test of SteadyRechargeShearStress model for marcc, saving output

Date: 1 April 2020
"""

import numpy as np

from landlab import RasterModelGrid
from DupuitLEM import SteadyRechargeShearStress
from DupuitLEM.grid_functions.grid_funcs import bind_avg_hydraulic_conductivity


#parameters
params = {}
params["recharge_rate"] = 1.5/(365*24*3600) #[m/s]
Ks = 1/3600 #[m/s]
K0 = 0.01/3600 #[m/s]
d_k = 1 #m
params["hydraulic_conductivity"] = bind_avg_hydraulic_conductivity(Ks,K0,d_k)
params["porosity"] = 0.2 #[]
params["regularization_factor"] = 0.01
params["courant_coefficient"] = 0.5
params["vn_coefficient"] = 0.8

params["permeability_production_rate"] = 2E-4/(365*24*3600) #[m/s]
params["characteristic_w_depth"] = 1 #m
params["uplift_rate"] = 1E-4/(365*24*3600) # uniform uplift [m/s]
params["b_st"] = 0.5 #shear stress erosion exponent
params["k_st"] = 5e-8 #shear stress erosion coefficient
params["shear_stress_threshold"] = 0.0 #threshold shear stress [N/m2]
params["manning_n"] = 0.05 #manning's n for flow depth calcualtion
params["hillslope_diffusivity"] = 0.01/(365*24*3600) # hillslope diffusivity [m2/s]

params["hydrological_timestep"] = 1e5 # hydrological timestep [s]
params["total_time"] = 1e5*(365*24*3600) # total simulation time [s]
params["morphologic_scaling_factor"] = 500 # morphologic scaling factor [-]

params["output_interval"] = 5000
params["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        "surface_water__discharge",
        "groundwater__specific_discharge_node",
        ]
params["base_output_path"] = './data/steady_ss_test_'
params["run_id"] = 0 #make this task_id if multiple runs


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

mdl = SteadyRechargeShearStress(params,save_output=True)

mdl.run_model()
