# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:15:41 2022

@author: dgbli
"""
#%%

import numpy as np
import pandas as pd
from itertools import product

from matplotlib import colors, ticker
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.colors import TwoSlopeNorm

from landlab.io.netcdf import from_netcdf, to_netcdf
from landlab import imshow_grid
import richdem as rd

directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'

#%%

i = 0

base_output_path = 'simple_lem_5_4'
lg_1 = np.array([15, 30, 60]) # geomorphic length scale [m]
hg_1 = np.array([2.25, 4.5, 9]) # geomorphic height scale [m]
lg_all = np.array(list(product(lg_1, hg_1)))[:,0]
hg_all = np.array(list(product(lg_1, hg_1)))[:,1]
lg = lg_all[i]
hg = hg_all[i]
tg = 22500 # geomorphic timescale [yr]
a0 = 0.7*15 #valley width factor [m]

# load grid
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
elev = grid.at_node['topographic__elevation']
elev_star = elev/hg
dx_star = grid.dx/lg
y = np.arange(grid.shape[0] + 1) * dx_star - dx_star * 0.5
x = np.arange(grid.shape[1] + 1) * dx_star - dx_star * 0.5
    
S = grid.at_node['slope_D8']
A = grid.at_node['drainage_area']
curvature = grid.at_node['curvature']
 
# make dimensionless versions
a_star = (A/grid.dx)/lg
S_star = S*lg/hg
steepness_star = np.sqrt(a_star)*S_star
curvature_star = curvature*lg**2/hg

# plot single hillshade
ls = LightSource(azdeg=135, altdeg=45)
plt.figure()
plt.imshow(ls.hillshade(elev_star.reshape(grid.shape).T, vert_exag=2, dx=dx_star, dy=dx_star), origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap='gray')
plt.ylabel(r'$y/\ell_g$')
plt.xlabel(r'$x/\ell_g$')
# plt.savefig('%s/hillshade_%s.png'%(save_directory, base_output_path), dpi=300) 

# plot slope-area and steepness-curvature
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
axs[0].scatter(a_star[grid.core_nodes], S_star[grid.core_nodes], alpha=0.2, s=5)
axs[0].set_xlabel("$a'$")
axs[0].set_ylabel("$S'$")
axs[0].set_xscale('log')
axs[0].set_yscale('log')
# plt.xlim((0.1,1e3))
x_plot = np.linspace(0,5, 10)
y_plot = x_plot - 1
axs[1].scatter(steepness_star[grid.core_nodes], curvature_star[grid.core_nodes], alpha=0.2, s=6)
axs[1].plot(x_plot,y_plot, 'k--', label=r"$\sqrt{a'} |\nabla' z'| - 1$")
axs[1].set_xlabel(r"$\sqrt{a'} |\nabla' z'|$")
axs[1].set_ylabel(r"$\nabla'^{2} z'$")
axs[1].legend(frameon=False)
plt.tight_layout()
# plt.savefig('%s/steep_curv_slope_area_%s.png'%(save_directory, base_output_path), dpi=300) 
# plt.savefig('pdf_figs/steep_curv_slope_area_%s.pdf'%base_output_path, dpi=300) 


#%%

base_output_path_simple = 'simple_lem_5_4'
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path_simple, i))
twi = np.log(grid.at_node['topographic__index_D8'])
twi[np.isinf(twi)] = np.nan
twi = twi[grid.core_nodes]

inds = np.argsort(twi)
twi_sort = twi[inds]
pr = (np.arange(len(twi_sort))+1)/len(twi_sort)

plt.figure()
plt.plot(twi_sort, pr)

#%%

base_output_path_simple = 'simple_lem_5_4'
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path_simple, 0))
z = grid.at_node['topographic__elevation']
z[grid.boundary_nodes] = np.nan
zrd = rd.rdarray(z.reshape(grid.shape), no_data=-9999)
zrd.geotransform = [0.0, grid.dx, 0.0, 0.0, 0.0, grid.dx]

profile_curvature = rd.TerrainAttribute(zrd, attrib='profile_curvature')
planform_curvature = rd.TerrainAttribute(zrd, attrib='planform_curvature')
curvature = rd.TerrainAttribute(zrd, attrib='curvature')
slope = rd.TerrainAttribute(zrd, attrib='slope_riserun')

slp = grid.add_zeros('node', "slope_rd")
slp[:] = slope.reshape(z.shape)    

pro = grid.add_zeros('node', "profile_curvature_rd")
pro[:] = profile_curvature.reshape(z.shape)

plan = grid.add_zeros('node', "planform_curvature_rd")
plan[:] = planform_curvature.reshape(z.shape)

curv = grid.add_zeros('node', "total_curvature_rd")
curv[:] = curvature.reshape(z.shape)

#%%
fig, ax = plt.subplots(figsize=(8,6))
field1 = grid.at_node["planform_curvature_rd"]
field2 = -grid.at_node["profile_curvature_rd"]
field3 = grid.at_node["total_curvature_rd"]


sc = ax.scatter(field1[grid.core_nodes],
              field2[grid.core_nodes],
              s=8,
              alpha=0.1,
              c=field3[grid.core_nodes],
              cmap='RdBu',
              norm=TwoSlopeNorm(vcenter=0.0),
              # vmin=0.0,
              #vmax=1.0,)
              )
# ax.set_xlim((-6,3)) #((-5, 1.5))
# ax.set_ylim((-1.5, 6)) #((-1, 4))
ax.axhline(0, linewidth=0.5 )
ax.axvline(0, linewidth=0.5 )
ax.set_xlabel(r"Planform Curvature")
ax.set_ylabel(r"Profile Curvature")
cbar = fig.colorbar(sc, orientation="vertical")
# cbar.set_label(label="P(sat)")
# cbar.solids.set_edgecolor("face")
# plt.savefig('%s/%s/curvatures_sat_scatter_%s_%d.png'%(directory, base_output_path, base_output_path, i), dpi=300)

#%% 

plan = grid.at_node["planform_curvature_rd"]
prof = -grid.at_node["profile_curvature_rd"]

curv_category = np.zeros_like(plan)
for i in range(len(plan)):
    if plan[i] > 0 and prof[i] > 0:
        curv_category[i] = 0
    elif plan[i] < 0 and prof[i] > 0:
        curv_category[i] = 1
    elif plan[i] < 0 and prof[i] < 0:
        curv_category[i] = 2
    elif plan[i] > 0 and prof[i] < 0:
        curv_category[i] = 3
    else:
        curv_category[i] = np.nan

lg = 15
hg = 2.25
dx = grid.dx/lg
y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
labels = ["Convex divergent", "Convex convergent", "Concave convergent", "Concave divergent" ]    

L2 = ['r', 'g', 'b', 'y']
cmap = colors.ListedColormap(L2)
norm = colors.BoundaryNorm(np.arange(-0.5, 4), cmap.N)
fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(curv_category.reshape(grid.shape).T, 
                     origin="lower", 
                     extent=(x[0], x[-1], y[0], y[-1]), 
                     cmap=cmap,
                     norm=norm,
                     )
fig.colorbar(im, format=fmt, ticks=np.arange(0,4))

plt.savefig('%s/%s/curv_zones_%s_%d.png'%(directory, base_output_path, base_output_path, i), dpi=300)
#%%
elev = grid.at_node['topographic__elevation']
elev_star = elev/hg
ls = LightSource(azdeg=135, altdeg=45)
plt.figure()
plt.imshow(ls.hillshade(elev_star.reshape(grid.shape).T, vert_exag=0.5, dx=dx, dy=dx_star), origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap='gray')
plt.ylabel(r'$y/\ell_g$')
plt.xlabel(r'$x/\ell_g$')