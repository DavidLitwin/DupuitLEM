
"""
test of SimpleSteadyRecharge model, without saving output

Date: 16 Mar 2020
"""

import numpy as np

from landlab import RasterModelGrid
from DupuitLEM import SimpleSteadyRecharge
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
params["m_sp"] = 0.5 #Exponent on Q []
params["n_sp"] = 1.0 #Exponent on S []
params["k_sp"] = 5e-8 #erosivity coefficient [m-1/2 s−1/2]
params["hillslope_diffusivity"] = 0.01/(365*24*3600) # hillslope diffusivity [m2/s]

params["hydrological_timestep"] = 1e5 # hydrological timestep [s]
params["total_time"] = 1e3*(365*24*3600) # total simulation time [s]
params["morphologic_scaling_factor"] = 500 # morphologic scaling factor [-]

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

mdl = SimpleSteadyRecharge(params,save_output=False)

mdl.run_model()
