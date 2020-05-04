
"""
test of StochasticRechargeShearStress model, without saving output

Date: 3 April 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

from landlab import RasterModelGrid
from landlab import imshow_grid
from landlab.io.netcdf import read_netcdf

from DupuitLEM import StochasticRechargeShearStress
from DupuitLEM.grid_functions.grid_funcs import bind_avg_hydraulic_conductivity


#parameters
params = {}
ID = 0
num = 8

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
params["total_hydrological_time"] = 365*24*3600 # total hydrological time
params["total_morphological_time"] = 1e4*(365*24*3600) # total simulation time [s]

params["precipitation_seed"] = 2
params["mean_storm_duration"] = 2*3600
params["mean_interstorm_duration"] = 48*3600
params["mean_storm_depth"] = 0.01

#load grid values from file
path = 'C:/Users/dgbli/Documents/MARCC_output/DupuitLEMResults/stoch_vary_k_'+str(num)+'-'+str(ID)+'/data/'
file = 'MSF500_stoch_vary_k_'+str(ID)+'_grid_60832.nc'
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

#run model
mdl.generate_exp_precip()
mdl.visualize_run_hydrological_step()

Q_all = mdl.Q_all[30:,:]
wtrel_all = mdl.wtrel_all[30:,:]

#proportion of time saturated
sat_all = np.greater_equal(wtrel_all, 0.99)
sat_time = np.sum(sat_all,axis=0)/np.shape(wtrel_all)[0]

plt.figure()
imshow_grid(grid,sat_time, cmap = 'Blues', limits=(0,1), colorbar_label = 'Proportion of time saturated', grid_units=('m','m'))
plt.savefig('C:/Users/dgbli/Documents/MARCC_output/DupuitLEMResults/figs/stoch_vary_k_'+str(num)+'/prop_sat_K=%.2f_d=%.2f.png'%(Ks*3600,d_k))


#amount of runoff from exfiltration and precip on sat area
time = mdl.time[30:]
qs_all = mdl.qs_all[30:,:]
p = mdl.intensity[30:]
exfilt_all = np.zeros_like(qs_all)
exfilt_sum = np.zeros(len(elev))
p_sat_all = np.zeros_like(qs_all)
p_sat_sum = np.zeros(len(elev))
for i in range(np.shape(qs_all)[0]-1):
    exfilt_all[i,:] = np.maximum(qs_all[i,:]-p[i],0)
    exfilt_sum += exfilt_all[i,:]*(time[i+1]-time[i])
    
    p_sat_all[i,:] = qs_all[i,:]-exfilt_all[i,:]
    p_sat_sum += p_sat_all[i,:]*(time[i+1]-time[i])
    
plt.figure()
imshow_grid(grid,exfilt_sum, cmap = 'Blues', colorbar_label = 'exfiltration', grid_units=('m','m'))
plt.savefig('C:/Users/dgbli/Documents/MARCC_output/DupuitLEMResults/figs/stoch_vary_k_'+str(num)+'/exfilt_K=%.2f_d=%.2f.png'%(Ks*3600,d_k))

plt.figure()
imshow_grid(grid,p_sat_sum, cmap = 'Blues', colorbar_label = 'precip on saturated area', grid_units=('m','m'))
plt.savefig('C:/Users/dgbli/Documents/MARCC_output/DupuitLEMResults/figs/stoch_vary_k_'+str(num)+'/p_sat_K=%.2f_d=%.2f.png'%(Ks*3600,d_k))



def plot_gif(source,index):
    
    fig = plt.figure(figsize=(8,6))
    imshow_grid(grid,source, cmap='gist_earth', limits=(0,50), colorbar_label = 'Elevation [m]', grid_units=('m','m'))
    plt.tight_layout()
    
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return image

imageio.mimsave('../DupuitLEMResults/figs/stoch_vary_k_'+str(num)+'/q_'+str(ID)+'.gif', [plot_gif(Q_all[i],i) for i in range(len(Q_all))], fps=1)



p_plot = intensity[intensity>0]
t_plot = time[intensity>0]

plt.figure()
bar = plt.bar(t_plot,p_plot,20000)
plt.show()



