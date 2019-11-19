# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:30:02 2019

@author: dgbli
"""

import numpy as np
import pickle

import matplotlib.pyplot as plt
from landlab import imshow_grid, RasterModelGrid
from landlab.io.netcdf import read_netcdf
from landlab.components import FlowAccumulator

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

from landlab.grid.mappers import map_max_of_node_links_to_node

#%%

grid_10000 = read_netcdf('./output/37205951/data/37205951_10000_grid.nc')
grid_100 = read_netcdf('./output/37205953/data/37205951_100_grid.nc')
grid_500 = read_netcdf('./output/37205954/data/37205951_500_grid.nc')
grid_1000 = read_netcdf('./output/37205955/data/37205951_1000_grid.nc')
grid_5000 = read_netcdf('./output/37205956/data/37205951_5000_grid.nc')

grid = grid_5000
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt = grid.at_node['water_table__elevation']

#%%

plt.figure(figsize=(8,6))
imshow_grid(grid,'topographic__elevation',cmap='gist_earth',colorbar_label = 'Elevation [m]', grid_units=('m','m'))
plt.savefig('./output/figs/elev_10000.png')

#%%
plt.figure(figsize=(8,6))
imshow_grid(grid,grid.at_node['topographic__elevation'] - grid.at_node['aquifer_base__elevation'],cmap='YlOrBr',colorbar_label = 'Regolith thickness [m]', grid_units=('m','m'))
plt.savefig('./output/figs/soil_10000.png')

#%%

plt.figure(figsize=(8,6))
imshow_grid(grid,(wt-base)/(elev-base),cmap='Blues',limits=(0,1),colorbar_label = 'Relative saturated thickness [-]', grid_units=('m','m'))
plt.savefig('./output/figs/rel_thickness_10000.png')


#%%

plt.figure(figsize=(8,6))
imshow_grid(grid,grid.at_node['surface_water__discharge'],cmap='plasma',colorbar_label = 'surface water discharge [m3/s]', grid_units=('m','m'))
plt.savefig('./output/figs/surface_water_10000.png')

#%%

y_node = [100,500,900]
fig,axs = plt.subplots(3,1,figsize=(8,6))
for i in range(3):
    middle_row = np.where(grid.x_of_node == y_node[2-i])[0][1:-1]
    
    axs[i].fill_between(grid.y_of_node[middle_row],elev[middle_row],base[middle_row],facecolor=colors['sienna'] )
    axs[i].fill_between(grid.y_of_node[middle_row],wt[middle_row],base[middle_row],facecolor=colors['royalblue'] )

axs[2].set_xlabel('Distance (m)')
axs[2].set_ylabel('Elevation (m)')
plt.savefig('./output/figs/cross_section_10000.png')

#%%

S_node = grid.calc_slope_at_node()
fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8',  \
                     depression_finder = 'DepressionFinderAndRouter')
fa.run_one_step()
area = grid.at_node['drainage_area']

plt.figure()
plt.loglog(area,S_node, '.')
plt.xlabel('area [m2]')
plt.ylabel('Slope [m/m]')
#plt.savefig('./Dupuit_LEM_results/slope_area_1570563142.540701.png')

