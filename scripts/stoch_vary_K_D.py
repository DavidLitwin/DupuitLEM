
"""
Use StochasticRechargeShearStress model to test different values stream
power coefficient and diffusivity. Save output. The goal of this test is to
identify whether the feature that appears for k_st = 1e10 and d=0.001
(bedrock amphitheater shaped channel heads) appears always or only in some cases.

Date: 10 May 2020
"""
import os
import numpy as np
from itertools import product

from landlab import RasterModelGrid
from DupuitLEM import StochasticRechargeShearStress
from DupuitLEM.grid_functions.grid_funcs import bind_avg_hydraulic_conductivity

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#parameters
params = {}
k_st_all_iter = np.array([1e-11, 5e-11, 1e-10])
D_all_iter = np.array([0.001, 0.005, 0.01])/(365*24*3600)

combinations = np.array(list(product(k_st_all_iter,D_all_iter)))
D_all = combinations[:,1]
k_st_all = combinations[:,0]

Ks = 0.5/3600 #[m/s]
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
params["k_st"] = k_st_all[ID] #shear stress erosion coefficient
params["shear_stress_threshold"] = 0.01 #threshold shear stress [N/m2]
params["chezy_c"] = 15 #chezy coefficient for flow depth calcualtion
params["hillslope_diffusivity"] = D_all[ID] # hillslope diffusivity [m2/s]

params["morphologic_scaling_factor"] = 500 # morphologic scaling factor [-]
params["total_hydrological_time"] = 30*24*3600 # total hydrological time
params["total_morphological_time"] = 2.5e6*(365*24*3600) # total simulation time [s]

params["precipitation_seed"] = 2
params["mean_storm_duration"] = 2*3600
params["mean_interstorm_duration"] = 48*3600
params["mean_storm_depth"] = 0.01

params["output_interval"] = 1000
params["output_fields"] = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        "surface_water__discharge",
        "groundwater__specific_discharge_node",
        ]
params["base_output_path"] = './data/MSF500_stoch_vary_k_'
params["run_id"] = ID #make this task_id if multiple runs

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
