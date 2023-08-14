# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:26:02 2020

@author: dgbli
"""

import numpy as np
import pyvista as pv
import pyvistaqt as pvq

from matplotlib import cm, colors, ticker
from landlab.io.netcdf import from_netcdf

directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path =  'steady_gam_sigma_1_stochev'

#%%
i = 4


grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
elev = grid.at_node['topographic__elevation']
sat_class = grid.at_node['saturation_class']
labels = ["dry", "variable", "wet"]
entropy = grid.at_node['saturation_entropy']
entropy[np.isnan(entropy)] = 0.0

elev_plot = elev.reshape(grid.shape)
cfield1 = sat_class.reshape(grid.shape)
cfield2 = entropy.reshape(grid.shape)

mx = grid.x_of_node[0:125]
my = grid.y_of_node[grid.x_of_node==0]

X,Y = np.meshgrid(mx,my)

grid_pv = pv.StructuredGrid(X,Y,elev_plot)

# grid_pv['Q'] = Q_plot

#%%

p = pvq.BackgroundPlotter()
p.background_color = 'white'
p.add_mesh(grid_pv,scalars=cfield2.T,cmap='plasma_r')
p.show()


p = pvq.BackgroundPlotter()
p.background_color = 'white'
p.add_mesh(grid_pv,scalars=elev_plot.T,cmap='gist_earth')
p.show()

#%%

L1 = ["peru", "dodgerblue", "navy"]
labels = ["dry", "variable", "wet"] 

p1 = pvq.BackgroundPlotter()
p1.add_mesh(grid_pv,scalars=elev_plot.T,cmap='viridis')
p1.show()


blue = np.array([12/256, 238/256, 246/256, 1])
yellow = np.array([255/256, 247/256, 0/256, 1])
grey = np.array([189/256, 189/256, 189/256, 1])

cmap = colors.ListedColormap(L1)
norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)


p = pvq.BackgroundPlotter()
p.background_color = 'white'
p.add_mesh(grid_pv,scalars=cfield1.T,cmap=cmap)
p.show()

#%%

blank_elev = 10*np.ones((100,100))
blank_elev[0,:] = 0
blank_elev[-1,:] = 0
blank_elev[:,0] = 0
blank_elev[:,-1] = 0


grid_pv = pv.StructuredGrid(X,Y,blank_elev)
p = pv.BackgroundPlotter()
p.background_color = 'white'
p.add_mesh(grid_pv,scalars=blank_elev.T,cmap='gist_earth')
p.show()

