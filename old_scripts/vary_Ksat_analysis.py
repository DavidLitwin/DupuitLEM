# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:04:20 2019

@author: dgbli
"""


import numpy as np
import imageio
import glob

import matplotlib.pyplot as plt
from landlab import imshow_grid, RasterModelGrid
from landlab.io.netcdf import read_netcdf
from landlab.components import FlowAccumulator

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

from landlab.grid.mappers import map_max_of_node_links_to_node

#%%

path_0 = './output/Ksat_test/attempt_3/37372038/data/'
path_2 = './output/Ksat_test/attempt_3/37372071/data/'
path_1 = './output/Ksat_test/attempt_3/37372072/data/'

grid_0 = read_netcdf('./output/Ksat_test/attempt_3/37372038/data/37372038_0.0005_grid_end.netcdf')
grid_2 = read_netcdf('./output/Ksat_test/attempt_3/37372071/data/37372038_5e-05_grid_end.netcdf')
grid_1 = read_netcdf('./output/Ksat_test/attempt_3/37372072/data/37372038_0.0001_grid_end.netcdf')

#%% Fixed Figures

grid = grid_2
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt = grid.at_node['water_table__elevation']



plt.figure(figsize=(8,6))
imshow_grid(grid,'topographic__elevation',cmap='gist_earth',colorbar_label = 'Elevation [m]', grid_units=('m','m'))
plt.savefig('./output/Ksat_test/attempt_3/figs/elev_0.00005.png')


plt.figure(figsize=(8,6))
imshow_grid(grid,grid.at_node['topographic__elevation'] - grid.at_node['aquifer_base__elevation'],cmap='YlOrBr',colorbar_label = 'Regolith thickness [m]', grid_units=('m','m'))
plt.savefig('./output/Ksat_test/attempt_3/figs/soil_0.00005.png')


plt.figure(figsize=(8,6))
imshow_grid(grid,(wt-base)/(elev-base),cmap='Blues',limits=(0,1),colorbar_label = 'Relative saturated thickness [-]', grid_units=('m','m'))
plt.savefig('./output/Ksat_test/attempt_3/figs/rel_thickness_0.00005.png')


plt.figure(figsize=(8,6))
imshow_grid(grid,grid.at_node['surface_water__discharge'],cmap='plasma',colorbar_label = 'surface water discharge [m3/s]', grid_units=('m','m'))
plt.savefig('./output/Ksat_test/attempt_3/figs/surface_water_0.00005.png')


y_node = [100,500,900]
fig,axs = plt.subplots(3,1,figsize=(8,6))
for i in range(3):
    middle_row = np.where(grid.x_of_node == y_node[2-i])[0][1:-1]
    
    axs[i].fill_between(grid.y_of_node[middle_row],elev[middle_row],base[middle_row],facecolor=colors['sienna'] )
    axs[i].fill_between(grid.y_of_node[middle_row],wt[middle_row],base[middle_row],facecolor=colors['royalblue'] )

axs[2].set_xlabel('Distance (m)')
axs[2].set_ylabel('Elevation (m)')
plt.savefig('./output/Ksat_test/attempt_3/figs/cross_section_0.00005.png')


S_node = grid.calc_slope_at_node()
fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8',  \
                     depression_finder = 'DepressionFinderAndRouter')
fa.run_one_step()
area = grid.at_node['drainage_area']

plt.figure()
plt.loglog(area,S_node, '.')
plt.xlabel('area [m2]')
plt.ylabel('Slope [m/m]')
plt.savefig('./output/Ksat_test/attempt_3/figs/slope_area_0.00005.png')


#%% GIFs: elevation


files = glob.glob(path_0+'*.nc')
files = sorted(files, key=lambda x:float(x[63:-3]))

indices = []
for i in range(len(files)):
    indices.append(files[i][63:-3])

def plot_topographic_elevation(file,index):
    
    grid = read_netcdf(file)
    
    fig = plt.figure(figsize=(8,6))
    imshow_grid(grid,'topographic__elevation',cmap='gist_earth',colorbar_label = 'Elevation [m]', limits = (0,30), grid_units=('m','m'))
    plt.title('i = ' + index)
    
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return image

kwargs_write = {'fps':2, 'quantizer':'nq'}
imageio.mimsave('./output/Ksat_test/attempt_3/figs/elev_0.00005.gif', [plot_topographic_elevation(files[i],indices[i]) for i in range(len(files))], fps=2)


#%% soil

files = glob.glob(path_2+'*.nc')
files = sorted(files, key=lambda x:float(x[63:-3]))

indices = []
for i in range(len(files)):
    indices.append(files[i][63:-3])

def plot_regolith_thickness(file,index):
    
    grid = read_netcdf(file)
    
    fig = plt.figure(figsize=(8,6))
    imshow_grid(grid,grid.at_node['topographic__elevation'] - grid.at_node['aquifer_base__elevation'],cmap='YlOrBr',colorbar_label = 'Regolith thickness [m]', limits=(0,10), grid_units=('m','m'))
    plt.title('i = ' + index)
    
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return image

kwargs_write = {'fps':2, 'quantizer':'nq'}
imageio.mimsave('./output/Ksat_test/attempt_3/figs/soil_0.00005.gif', [plot_regolith_thickness(files[i],indices[i]) for i in range(len(files))], fps=2)

#%% maximum relative change plots

max_change_0 = np.loadtxt(path_0+'37372038_0.0005_max_rel_change.txt')
max_change_1 = np.loadtxt(path_1+'37372038_0.0001_max_rel_change.txt')
max_change_2 = np.loadtxt(path_2+'37372038_5e-05_max_rel_change.txt')

plt.figure(figsize=(8,6))
plt.plot(max_change_0,label='K=0.0005')
plt.plot(max_change_1,label='K=0.0001')
plt.plot(max_change_2,label='K=0.00005')
plt.ylabel('Maximum relative elevation change')
plt.xlabel('Time step')
plt.legend()
plt.savefig('./output/Ksat_test/attempt_3/figs/rel_change.png')

plt.figure(figsize=(8,6))
plt.semilogy(max_change_0,label='K=0.0005')
plt.semilogy(max_change_1,label='K=0.0001')
plt.semilogy(max_change_2,label='K=0.00005')
plt.ylabel('Maximum relative elevation change')
plt.xlabel('Time step')
plt.legend()
plt.savefig('./output/Ksat_test/attempt_3/figs/log_rel_change.png')



