
"""
test of StochasticRechargeShearStress model, without saving output

Date: 3 April 2020
"""

import numpy as np
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab import imshow_grid
from landlab.io.netcdf import read_netcdf

from DupuitLEM import StochasticRechargeShearStress
from DupuitLEM.grid_functions.grid_funcs import bind_avg_hydraulic_conductivity


#parameters
params = {}
ID = 0
num = 4

Ks_all = np.array([0.01, 0.05, 0.1, 0.5, 1.0])*(1/3600) #[m/s]
Ks = Ks_all[ID]
K0 = 0.01*Ks # asymptotic hydraulic conductivity at infinite depth
d_k = 1 #m
params["hydraulic_conductivity"] = bind_avg_hydraulic_conductivity(Ks,K0,d_k)
params["porosity"] = 0.2 #[]
params["regularization_factor"] = 0.01
params["courant_coefficient"] = 0.5
params["vn_coefficient"] = 0.8

params["permeability_production_rate"] = 2E-4/(365*24*3600) #[m/s]
params["characteristic_w_depth"] = 1 #m
params["uplift_rate"] = 1E-4/(365*24*3600) # uniform uplift [m/s]
params["b_st"] = 1.5 #shear stress erosion exponent
params["k_st"] = 1e-10 #shear stress erosion coefficient
params["shear_stress_threshold"] = 0.01 #threshold shear stress [N/m2]
params["manning_n"] = 0.05 #manning's n for flow depth calcualtion
params["hillslope_diffusivity"] = 0.01/(365*24*3600) # hillslope diffusivity [m2/s]

params["morphologic_scaling_factor"] = 500 # morphologic scaling factor [-]
params["total_hydrological_time"] = 30*24*3600 # total hydrological time
params["total_morphological_time"] = 1e4*(365*24*3600) # total simulation time [s]

params["precipitation_seed"] = 2
params["mean_storm_duration"] = 2*3600
params["mean_interstorm_duration"] = 48*3600
params["mean_storm_depth"] = 0.01

#load grid values from file
path = 'C:/Users/dgbli/Documents/MARCC_output/DupuitLEMResults/stoch_vary_k_'+str(num)+'-'+str(ID)+'/data/'
file = 'MSF500_stoch_vary_k_'+str(ID)+'_grid_12165.nc'
path1 = path+file
mg = read_netcdf(path1)

grid = RasterModelGrid((100, 100), xy_spacing=10.0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
base = grid.add_zeros('node', 'aquifer_base__elevation')
wt = grid.add_zeros('node', 'water_table__elevation')
    
elev[:] = mg.at_node['topographic__elevation']
base[:] = mg.at_node['aquifer_base__elevation']
wt[:] = mg.at_node['water_table__elevation']

params["grid"] = grid

mdl = StochasticRechargeShearStress(params,save_output=False,verbose=True)

#%%


mdl.generate_exp_precip()


time, intensity, tau_all, Q_all, wtrel_all = mdl.visualize_run_hydrological_step()


Q_range = np.max(Q_all,axis=0) - np.min(Q_all,axis=0)
wt_range = np.max(wtrel_all,axis=0) - np.min(wtrel_all,axis=0)


plt.figure()
imshow_grid(grid,Q_range)

plt.figure()
imshow_grid(grid,np.min(Q_all,axis=0))


plt.figure()
imshow_grid(grid,wt_range)




