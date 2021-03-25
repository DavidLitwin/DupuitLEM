# -*- coding: utf-8 -*-
"""

Plots for "Groundwater affects geomorphic and hydrologic properties of coevolved landscapes"
 
@author: David Litwin

Each section below contains the template for a figure in the manuscript, in 
order of their appearance. Model run codes are listed below. A complete table
of parameters for all model runs is included in a separate file.


List of model runs used in manuscript:
	- simple_lem_5_5(0-8): NoHyd vary hg-lg figures
	- steady_sp_5_13(0-8): low lambda vary hg-lg figure
	- steady_sp_5_14(0-8): high lambda vary hg-lg figure
	- steady_sp_3_15(0-39): gamma-lambda for hillslope length figure (med alpha)
	- steady_sp_3_16(0-39): gamma-lambda for hillslope length figure (low alpha)
	- steady_sp_3_17(0-39): gamma-lambda for hillslope length figure (high alpha)
	- steady_sp_3_18(0-29): gamma-lambda main figures
	- steady_sp_3_19(0-29): gamma-lambda supplemental (high alpha)
	- steady_sp_3_20(0-29): gamma-lambda supplemental (low alpha)
"""

import copy
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource, TwoSlopeNorm
import matplotlib.colors as colors
from matplotlib import cm
plt.rc('text', usetex=True)

from landlab.io.netcdf import from_netcdf

# directories
save_directory = 'C:/Users/dgbli/Documents/Papers/Ch1_groundwater_landscape_evolution/figs'
directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'

#%% NoHyd: scaling with varying hg and lg

plt.rc('text', usetex=True)
ls = LightSource(azdeg=135, altdeg=45)
fig = plt.figure(figsize=(7.5,7.5))
gs = GridSpec(3, 3, figure=fig)
subplots = {}
subplots[0] = fig.add_subplot(gs[2, 0])
subplots[1] = fig.add_subplot(gs[1, 0])
subplots[2] = fig.add_subplot(gs[0, 0])
subplots[3] = fig.add_subplot(gs[2, 1])
subplots[4] = fig.add_subplot(gs[1, 1])
subplots[6] = fig.add_subplot(gs[2, 2])
subplots[8] = fig.add_subplot(gs[0, 2])

base_output_path = 'simple_lem_5_5'
df_params = pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb'))
# lg_1 = np.array([15, 30, 60]) # geomorphic length scale [m]
# hg_1 = np.array([2.25, 4.5, 9]) # geomorphic height scale [m]
# lg_all = np.array(list(product(lg_1, hg_1)))[:,0]
# hg_all = np.array(list(product(lg_1, hg_1)))[:,1]
names = ['i', 'iv', 'vi', 'ii', 'v', '', 'iii', '', 'vii']
elev_star_all = []
ticks = range(0,100, 25)

for i, ax in subplots.items():

    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
            
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    elev_star = elev/hg
    elev_star_all.append(elev_star)
    dx_star = grid.dx/lg
    y = np.arange(grid.shape[0] + 1) * dx_star - dx_star * 0.5
    x = np.arange(grid.shape[1] + 1) * dx_star - dx_star * 0.5
    
    ax.imshow(ls.hillshade(elev_star.reshape(grid.shape).T, vert_exag=2, dx=dx_star, dy=dx_star), origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap='gray')
    ax.plot((min(grid.y_of_node)/lg,max(grid.y_of_node)/lg),(np.median(grid.x_of_node)/lg,np.median(grid.x_of_node)/lg), 'r--', alpha=0.85 )
    ax.set_title(r'$\alpha$=%.3f'%(hg/lg))
    ax.text(0.03, 
            0.97, 
            names[i], 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(ec='w',fc='w')
            ) 
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if i%3 != 0:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$x/\ell_g$ \newline \vspace{12mm} {\LARGE $\ell_g=%d$ m}'%lg)
    if i>2:
        ax.set_yticklabels([])
    else: 
        ax.set_ylabel(r'\centering {\LARGE $h_g=%.2f$ m} \newline \vspace{12mm} $y/\ell_g$'%hg)
        
plt.savefig('%s/hs_hg_lg_simple_lem.png'%save_directory, dpi=300)


#%% NoHyd: scaling with varying hg and lg cross sections

plt.rc('text', usetex=True)
ls = LightSource(azdeg=135, altdeg=45)
fig = plt.figure(figsize=(7.5,7.5))
gs = GridSpec(3, 3, figure=fig)
subplots = {}
subplots[0] = fig.add_subplot(gs[2, 0])
subplots[1] = fig.add_subplot(gs[1, 0])
subplots[2] = fig.add_subplot(gs[0, 0])
subplots[3] = fig.add_subplot(gs[2, 1])
subplots[4] = fig.add_subplot(gs[1, 1])
subplots[6] = fig.add_subplot(gs[2, 2])
subplots[8] = fig.add_subplot(gs[0, 2])

base_output_path = 'simple_lem_5_5'
df_params = pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb'))
# lg_1 = np.array([15, 30, 60]) # geomorphic length scale [m]
# hg_1 = np.array([2.25, 4.5, 9]) # geomorphic height scale [m]
# lg_all = np.array(list(product(lg_1, hg_1)))[:,0]
# hg_all = np.array(list(product(lg_1, hg_1)))[:,1]
names = ['i', 'iv', 'vi', 'ii', 'v', '', 'iii', '', 'vii']
elev_star_all = []
xticks = range(0,100, 25)
yticks = range(0,25,5)

for i, ax in subplots.items():

    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
        
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    middle_row = np.where(grid.x_of_node == np.median(grid.x_of_node))[0][1:-1]
    
    y = grid.y_of_node[middle_row]/lg
    ax.fill_between(y,elev[middle_row]/hg,np.zeros_like(elev[middle_row]),facecolor=(111/256,111/256,111/256))
    
    ax.set_title(r'$\alpha$=%.3f'%(hg/lg))
    ax.text(0.04, 
            0.96, 
            names[i], 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(ec='w',fc='w')
            ) 
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim(min(yticks), max(yticks))
    if i%3 != 0:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$x/\ell_g$ \newline \vspace{12mm} {\LARGE $\ell_g=%d$ m}'%lg)
    if i>2:
        ax.set_yticklabels([])
    else: 
        ax.set_ylabel(r'\centering {\LARGE $h_g=%.2f$ m} \newline \vspace{12mm} $z/h_g$'%hg)
        
plt.savefig('%s/hs_hg_lg_simple_lem_XS.png'%save_directory, dpi=300)


#%% DupuitLEM lambda=5: scaling with varying hg and lg

ls = LightSource(azdeg=135, altdeg=45)
fig = plt.figure(figsize=(7.5, 7.5))
gs = GridSpec(3, 3, figure=fig)
subplots = {}
subplots[0] = fig.add_subplot(gs[2, 0])
subplots[1] = fig.add_subplot(gs[1, 0])
subplots[2] = fig.add_subplot(gs[0, 0])
subplots[3] = fig.add_subplot(gs[2, 1])
subplots[4] = fig.add_subplot(gs[1, 1])
subplots[6] = fig.add_subplot(gs[2, 2])
subplots[8] = fig.add_subplot(gs[0, 2])

base_output_path = 'steady_sp_5_14'
df_params = pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb'))
lg_all = df_params['lg']
hg_all = df_params['hg']
names = ['i', 'iv', 'vi', 'ii', 'v', '', 'iii', '', 'vii']
elev_star_all = []
ticks = range(0,100, 25)

for i, ax in subplots.items():

    lg = lg_all[i] # geomorphic length scale [m]
    hg = hg_all[i] # geomorphic height scale [m]
        
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    elev_star = elev/hg
    elev_star_all.append(elev_star)
    dx_star = grid.dx/lg
    y = np.arange(grid.shape[0] + 1) * dx_star - dx_star * 0.5
    x = np.arange(grid.shape[1] + 1) * dx_star - dx_star * 0.5
    
    ax.imshow(ls.hillshade(elev_star.reshape(grid.shape).T, vert_exag=2, dx=dx_star, dy=dx_star), origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap='gray')
    ax.plot((min(grid.y_of_node)/lg,max(grid.y_of_node)/lg),(np.median(grid.x_of_node)/lg,np.median(grid.x_of_node)/lg), 'r--', alpha=0.85 )
    ax.set_title(r'$\alpha$=%.3f'%(hg/lg))
    ax.text(0.03, 
            0.97, 
            names[i], 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(ec='w',fc='w')
            ) 
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if i%3 != 0:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$x/\ell_g$ \newline \vspace{12mm} {\LARGE $\ell_g=%d$ m}'%lg)
    if i>2:
        ax.set_yticklabels([])
    else: 
        ax.set_ylabel(r'\centering {\LARGE $h_g=%.2f$ m} \newline \vspace{12mm} $y/\ell_g$'%hg)
        
plt.savefig('%s/hs_hg_lg_steady_sp.png'%save_directory, dpi=300)

#%% DupuitLEM lambda=5: scaling with varying hg and lg cross sections

ls = LightSource(azdeg=135, altdeg=45)
fig = plt.figure(figsize=(7.5, 7.5))
gs = GridSpec(3, 3, figure=fig)
subplots = {}
subplots[0] = fig.add_subplot(gs[2, 0])
subplots[1] = fig.add_subplot(gs[1, 0])
subplots[2] = fig.add_subplot(gs[0, 0])
subplots[3] = fig.add_subplot(gs[2, 1])
subplots[4] = fig.add_subplot(gs[1, 1])
subplots[6] = fig.add_subplot(gs[2, 2])
subplots[8] = fig.add_subplot(gs[0, 2])

base_output_path = 'steady_sp_5_14'
df_params = pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb'))
zmax = 180
zmin = -10

names = ['i', 'iv', 'vi', 'ii', 'v', '', 'iii', '', 'vii']
elev_star_all = []
xticks = range(0,100, 25)
yticks = range(0,200, 50)

for i, ax in subplots.items():

    lg = df_params['lg'][i] # geomorphic length scale [m]
    hg = df_params['hg'][i] # geomorphic height scale [m]
    b =  df_params['b'][i]

    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation'] - b
    base = grid.at_node['aquifer_base__elevation'] - b
    wt = grid.at_node['water_table__elevation'] - b
    middle_row = np.where(grid.x_of_node == np.median(grid.x_of_node))[0][1:-1]
    
    y = grid.y_of_node[middle_row]/lg
    ax.fill_between(y,elev[middle_row]/hg,base[middle_row]/hg,facecolor=(198/256,155/256,126/256) )
    ax.fill_between(y,wt[middle_row]/hg,base[middle_row]/hg,facecolor=(145/256,176/256,227/256))
    ax.fill_between(y,base[middle_row]/hg,zmin*np.ones_like(base[middle_row]),facecolor=(111/256,111/256,111/256))
    ax.set_ylim((zmin,zmax))
    
    ax.set_title(r'$\alpha$=%.3f'%(hg/lg))
    ax.text(0.04, 
            0.96, 
            names[i], 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(ec='w',fc='w')
            ) 
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # ax.set_xlim(min(xticks), max(xticks))
    # ax.set_ylim(min(yticks), 180)
    if i%3 != 0:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$x/\ell_g$ \newline \vspace{12mm} {\LARGE $\ell_g=%d$ m}'%lg)
    if i>2:
        ax.set_yticklabels([])
    else: 
        ax.set_ylabel(r'\centering {\LARGE $h_g=%.2f$ m} \newline \vspace{12mm} $z/h_g$'%hg)
        
    if i==0:
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        y = grid.y_of_node[middle_row]/lg
        axins.fill_between(y,elev[middle_row]/hg,base[middle_row]/hg,facecolor=(198/256,155/256,126/256) )
        axins.fill_between(y,wt[middle_row]/hg,base[middle_row]/hg,facecolor=(145/256,176/256,227/256))
        axins.fill_between(y,base[middle_row]/hg,np.zeros_like(base[middle_row]),facecolor=(111/256,111/256,111/256))
        # sub region of the original image
        x1, x2, y1, y2 = 33, 42, 40, 46
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')

        ax.indicate_inset_zoom(axins)
        
plt.savefig('%s/hs_hg_lg_steady_sp_XS.png'%save_directory, dpi=300)

#%% DupuitLEM lambda=0.01: scaling with varying hg and lg

ls = LightSource(azdeg=135, altdeg=45)
fig = plt.figure(figsize=(7.5, 7.5))
gs = GridSpec(3, 3, figure=fig)
subplots = {}
subplots[0] = fig.add_subplot(gs[2, 0])
subplots[1] = fig.add_subplot(gs[1, 0])
subplots[2] = fig.add_subplot(gs[0, 0])
subplots[3] = fig.add_subplot(gs[2, 1])
subplots[4] = fig.add_subplot(gs[1, 1])
subplots[6] = fig.add_subplot(gs[2, 2])
subplots[8] = fig.add_subplot(gs[0, 2])

base_output_path = 'steady_sp_5_13'
df_params = pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb'))
lg_all = df_params['lg']
hg_all = df_params['hg']
names = ['i', 'iv', 'vi', 'ii', 'v', '', 'iii', '', 'vii']
elev_star_all = []
ticks = range(0,150, 25)

for i, ax in subplots.items():

    lg = lg_all[i] # geomorphic length scale [m]
    hg = hg_all[i] # geomorphic height scale [m]
        
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    elev_star = elev/hg
    elev_star_all.append(elev_star)
    dx_star = grid.dx/lg
    y = np.arange(grid.shape[0] + 1) * dx_star - dx_star * 0.5
    x = np.arange(grid.shape[1] + 1) * dx_star - dx_star * 0.5
    
    ax.imshow(ls.hillshade(elev_star.reshape(grid.shape).T, vert_exag=2, dx=dx_star, dy=dx_star), origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap='gray')
    ax.plot((min(grid.y_of_node)/lg,max(grid.y_of_node)/lg),(np.median(grid.x_of_node)/lg,np.median(grid.x_of_node)/lg), 'r--', alpha=0.85 )
    ax.set_title(r'$\alpha$=%.3f'%(hg/lg))
    ax.text(0.03, 
            0.97, 
            names[i], 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(ec='w',fc='w')
            ) 
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if i%3 != 0:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$x/\ell_g$ \newline \vspace{12mm} {\LARGE $\ell_g=%d$ m}'%lg)
    if i>2:
        ax.set_yticklabels([])
    else: 
        ax.set_ylabel(r'\centering {\LARGE $h_g=%.2f$ m} \newline \vspace{12mm} $y/\ell_g$'%hg)
        
plt.savefig('%s/hs_hg_lg_steady_sp_low_lam.png'%save_directory, dpi=300)

#%% DupuitLEM lambda=0.01: scaling with varying hg and lg cross sections

ls = LightSource(azdeg=135, altdeg=45)
fig = plt.figure(figsize=(7.5, 7.5))
gs = GridSpec(3, 3, figure=fig)
subplots = {}
subplots[0] = fig.add_subplot(gs[2, 0])
subplots[1] = fig.add_subplot(gs[1, 0])
subplots[2] = fig.add_subplot(gs[0, 0])
subplots[3] = fig.add_subplot(gs[2, 1])
subplots[4] = fig.add_subplot(gs[1, 1])
subplots[6] = fig.add_subplot(gs[2, 2])
subplots[8] = fig.add_subplot(gs[0, 2])

base_output_path = 'steady_sp_5_13'
df_params = pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb'))
zmax = 600
zmin = -300

names = ['i', 'iv', 'vi', 'ii', 'v', '', 'iii', '', 'vii']
elev_star_all = []
xticks = range(0,200, 50)
yticks = range(zmin,zmax+300, 300)

for i, ax in subplots.items():

    lg = df_params['lg'][i] # geomorphic length scale [m]
    hg = df_params['hg'][i] # geomorphic height scale [m]
    b =  df_params['b'][i]

    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation'] - b
    base = grid.at_node['aquifer_base__elevation'] - b
    wt = grid.at_node['water_table__elevation'] - b
    middle_row = np.where(grid.x_of_node == np.median(grid.x_of_node))[0][1:-1]
    
    y = grid.y_of_node[middle_row]/lg
    ax.fill_between(y,elev[middle_row]/hg,base[middle_row]/hg,facecolor=(198/256,155/256,126/256) )
    ax.fill_between(y,wt[middle_row]/hg,base[middle_row]/hg,facecolor=(145/256,176/256,227/256))
    ax.fill_between(y,base[middle_row]/hg,zmin*np.ones_like(base[middle_row]),facecolor=(111/256,111/256,111/256))
    ax.set_ylim((zmin,zmax))
    
    ax.set_title(r'$\alpha$=%.3f'%(hg/lg))
    ax.text(0.04, 
            0.96, 
            names[i], 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(ec='w',fc='w')
            ) 
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # ax.set_ylim(min(yticks), max(yticks))
    if i%3 != 0:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r'$x/\ell_g$ \newline \vspace{12mm} {\LARGE $\ell_g=%d$ m}'%lg)
    if i>2:
        ax.set_yticklabels([])
    else: 
        ax.set_ylabel(r'\centering {\LARGE $h_g=%.2f$ m} \newline \vspace{12mm} $z/h_g$'%hg)

        
plt.savefig('%s/hs_hg_lg_steady_sp_low_lam_XS.png'%save_directory, dpi=300)

#%% DupuitLEM: vary gamma and lambda, load results

# specify model runs
base_output_path = 'steady_sp_3_18'
model_runs = np.arange(30)

# load params
df_params = pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb'))

# load results
dfs = []
for ID in model_runs:
    
    d = pickle.load(open('%s/%s/output_ID_%d.p'%(directory,base_output_path, ID), 'rb'))
    df = pd.DataFrame([d])
    dfs.append(df)
df_results = pd.concat(dfs, axis=0, ignore_index=True)

#%% DupuitLEM: vary gamma and lambda, hillshades

plot_runs = model_runs
nrows = 6
ncols = 5
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

# plot_runs hillshades:
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,9))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    # grid = read_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    ls = LightSource(azdeg=135, altdeg=45)
    axs[m,n].imshow(
                    ls.hillshade(elev.reshape(grid.shape).T, 
                        vert_exag=2, 
                        dx=grid.dx, 
                        dy=grid.dy), 
                    origin="lower", 
                    extent=(x[0], x[-1], y[0], y[-1]), 
                    cmap='gray',
                    )
    axs[m,n].text(0.05, 
                0.95, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                )   
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])

axs[-1, 0].set_ylabel(r'$y/\ell_g$')
axs[-1, 0].set_xlabel(r'$x/\ell_g$')
plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=None, wspace=0.15, hspace=0.15)
plt.savefig('%s/hillshade_%s.png'%(save_directory, base_output_path), dpi=300)


#%% DupuitLEM: vary gamma and lambda, cross sections

zmax_gam = np.flip(np.array([15, 15, 20, 45, 120, 240]))
zmin_gam = np.flip(np.array([-8, -12, -15, -20, -40, -100]))

plot_runs = model_runs
nrows = 6
ncols = 5
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

# plot_runs hillshades:
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,9))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    hg = df_params['hg'][i] # all the same hg and lg
    lg = df_params['lg'][i]
    b = df_params['b'][i]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation'] - b
    base = grid.at_node['aquifer_base__elevation'] - b
    wt = grid.at_node['water_table__elevation'] - b
    middle_row = np.where(grid.x_of_node == np.median(grid.x_of_node))[0][1:-1]
    
    y = grid.y_of_node[middle_row]/lg
    axs[m,n].fill_between(y,elev[middle_row]/hg,base[middle_row]/hg,facecolor=(198/256,155/256,126/256) )
    axs[m,n].fill_between(y,wt[middle_row]/hg,base[middle_row]/hg,facecolor=(145/256,176/256,227/256))
    axs[m,n].fill_between(y,base[middle_row]/hg,zmin_gam[m]*np.ones_like(base[middle_row]),facecolor=(111/256,111/256,111/256))
    axs[m,n].set_ylim((zmin_gam[m],zmax_gam[m]))
    # ax.set_xlim((0,995))
    
    axs[m,n].set_xticks(range(0,100,25))
    if m != nrows-1:
        axs[m, n].set_xticklabels([])

axs[-1, 0].set_ylabel(r'$z/h_g$')
axs[-1, 0].set_xlabel(r'$x/\ell_g$')
plt.tight_layout()
plt.savefig('%s/cross_section_%s.png'%(save_directory, base_output_path), dpi=300)


#%% DupuitLEM: vary gamma and lambda, map view qstar

plot_runs = model_runs
nrows = 6
ncols = 5
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

max_steep=0
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,7))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    qstar = grid.at_node['qstar']
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 

    im = axs[m, n].imshow(qstar.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap='plasma_r',
                         vmin=0.0,
                         vmax=1.0,
                         )
    axs[m,n].text(0.05, 
                0.95, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=8, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                )  
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    
plt.subplots_adjust(left=0.15, right=0.8, wspace=0.15, hspace=0.15)
# plt.tight_layout()
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) #Left, Bottom, Width, Height
fig.colorbar(im, cax=cbar_ax, label=r"$Q^*$")
axs[-1, 0].set_ylabel(r'$y/\ell_g$')
axs[-1, 0].set_xlabel(r'$x/\ell_g$')
plt.savefig('%s/qstar_%s.png'%(save_directory, base_output_path), dpi=300)

#%% DupuitLEM: vary gamma and lambda, CDF of Q*

plot_runs=model_runs
nrows=6
ncols=5
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)


fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4,4))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    qstar = grid.at_node['qstar']
    
    Ep = np.arange(grid.number_of_core_nodes)/(1+grid.number_of_core_nodes)
    Qstar = np.sort(qstar[grid.core_nodes])
    
    
    axs[m, n].plot(Qstar, Ep)
    # axs[m, n].text(0.05, 
    #                 0.25, 
    #                 str(i), 
    #                 transform=axs[m, n].transAxes, 
    #                 fontsize=8, 
    #                 verticalalignment='top',
    #                 )
    axs[m, n].tick_params(axis='both', which='major', labelsize=6)
    axs[m, n].set_xlim((0, 1.0))
    axs[m, n].set_ylim((0, 1.05))

    if m%nrows != nrows-1:
        axs[m, n].set_xticklabels([])
    if n%ncols != 0:
        axs[m, n].set_yticklabels([])
    
plt.subplots_adjust(wspace=0.35, hspace=0.2)
# fig.suptitle('Hydrologic Balance', fontsize=16, y=0.92)
axs[-1, 0].set_ylabel('CDF')
axs[-1, 0].set_xlabel('$Q^*$')
# axs[-1, 0].legend(frameon=False, loc=4)
plt.savefig('%s/qstar_cdf_%s.png'%(save_directory, base_output_path), dpi=300)

#%% DupuitLEM: vary gamma and lambda, Q* and curvature probabilities

viridis = cm.get_cmap('viridis', 5)
col = viridis.colors

lam_all = pd.unique(df_params['lam'])
gam_all = pd.unique(df_params['gam'])
n = len(gam_all)

qstar_med = np.zeros(len(df_params))
qstar_0 = np.zeros(len(df_params))
curv_0 = np.zeros(len(df_params))
for i in range(len(df_params)):
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    qstar = grid.at_node['qstar']
    curv = grid.at_node['curvature']
    qstar_med[i] = np.sum(qstar[grid.core_nodes]>0.5)/grid.number_of_core_nodes
    qstar_0[i] = np.sum(qstar[grid.core_nodes]>0.001)/grid.number_of_core_nodes
    curv_0[i] = np.sum(curv[grid.core_nodes]>0.0)/grid.number_of_core_nodes
    
styles = [':', '--', '-']
fig, ax = plt.subplots(figsize=(4,3))
for i in range(len(lam_all)):
    ax.plot(gam_all, curv_0[n*i:n*i+n], color=col[i], linestyle=styles[0]) #label=r'$\lambda$=%.2f'%lam_all[i])
    ax.scatter(gam_all, curv_0[n*i:n*i+n], color=col[i])
    
    ax.plot(gam_all, qstar_med[n*i:n*i+n], color=col[i], linestyle=styles[1])
    ax.scatter(gam_all, qstar_med[n*i:n*i+n], color=col[i])
    
    ax.plot(gam_all, qstar_0[n*i:n*i+n], color=col[i], linestyle=styles[2], label=r'$\lambda$=%.2f'%lam_all[i])
    ax.scatter(gam_all, qstar_0[n*i:n*i+n], color=col[i])
    
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$', fontsize=14)
ax.set_ylabel(r'$P(\cdot)$')
legend0 = plt.legend(frameon=False, loc=(0.63,0.45))

# dummy lines to make a legend of different linestyles, independent of lambda legend
dummy_lines = []
for style in styles:
    dummy_lines.append(ax.plot([],[], c="black", ls = style)[0])
legend1 = plt.legend([dummy_lines[i] for i in [0,1,2]], [r'$\nabla^2 z >$0', r'$Q^*>$0.5', r'$Q^*>$0.001'], loc=(0.63,0.15), frameon=False) #[r'$\nabla z >0$', r'$Q^*>0.5$', r'$Q^*>0.001&'])
# ax.set_ylim((0.1, 30))
ax.add_artist(legend0)


plt.tight_layout()
plt.savefig('%s/qstar_cond_%s.png'%(save_directory, base_output_path), dpi=300)


#%% NoHyd: slope-area and steepness-curvature

i = 0

base_output_path = 'simple_lem_5_5'
df_params = pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb'))
lg = df_params['lg'][i]
hg = df_params['hg'][i]

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
plt.savefig('%s/hillshade_%s.png'%(save_directory, base_output_path), dpi=300) 

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
plt.savefig('%s/steep_curv_slope_area_%s.png'%(save_directory, base_output_path), dpi=300) 


#%% DupuitLEM: map view steepness

nrows=3
ncols=2
plot_runs = np.array([0,2,5,24,26,29])
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

max_steep=0
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    S = grid.at_node['slope_D8']
    A = grid.at_node['drainage_area']
    a_star = (A/grid.dx)/lg
    S_star = S*lg/hg
    steepness_star = np.sqrt(a_star)*S_star
    
    max_steep = max(max_steep,max(steepness_star))

    im = axs[m, n].imshow(steepness_star.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap='viridis',
                         norm = colors.LogNorm(vmin=1, vmax=90)
                         )
    axs[m,n].set_title(r"$\gamma=%.2f$, $\lambda=%.2f$"%(df_params['gam'][i], df_params['lam'][i]), fontsize=9)
    axs[m,n].text(0.04, 
                0.96, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=8, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                ) 
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    
plt.subplots_adjust(left=0.15, right=0.8, wspace=0.15, hspace=0.15)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) #Left, Bottom, Width, Height
fig.colorbar(im, cax=cbar_ax, label=r"$\sqrt{a'} |\nabla' z'|$")
axs[-1, 0].set_ylabel(r'$y/\ell_g$')
axs[-1, 0].set_xlabel(r'$x/\ell_g$')
plt.savefig('%s/steepness_%s.png'%(save_directory, base_output_path), dpi=300)

#%% DupuitLEM: map view curvature

nrows=3
ncols=2
plot_runs = np.array([0,2,5,24,26,29])
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

max_curv=0
min_curv=0
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    curvature = grid.at_node['curvature']
    curvature_star = curvature*lg**2/hg
    
    max_curv = max(max_curv,max(curvature_star))
    min_curv = min(min_curv,min(curvature_star))

    im = axs[m, n].imshow(curvature_star.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap='RdBu',
                         norm=TwoSlopeNorm(0.0, vmin=-1.1)
                         )
    fig.colorbar(im, ax=axs[m,n], label=r"$\nabla'^{2} z'$", fraction=0.046, pad=0.04)
    axs[m,n].set_title(r"$\gamma=%.2f$, $\lambda=%.2f$"%(df_params['gam'][i], df_params['lam'][i]), fontsize=9)
    axs[m,n].text(0.05, 
                0.95, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                ) 

    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    
plt.subplots_adjust(wspace=0.35, hspace=0.15, right=0.85)
axs[-1, 0].set_ylabel(r'$y/\ell_g$')
axs[-1, 0].set_xlabel(r'$x/\ell_g$')
plt.savefig('%s/curvature_%s.png'%(save_directory, base_output_path), dpi=300)

#%% DupuitLEM: map view topographic index

nrows=3
ncols=2
plot_runs = np.array([0,2,5,24,26,29])
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

max_twi=0
min_twi=0
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    S4 = grid.at_node['slope_D4']
    twi = grid.at_node['topographic__index_D4']/np.cos(np.arctan(S4))**2
    twi_star = twi*hg/lg**2
    
    max_twi = max(max_twi,np.nanmax(twi_star[np.isfinite(twi_star)]))
    min_twi = min(min_twi,np.nanmin(twi_star[np.isfinite(twi_star)]))
    
    im = axs[m, n].imshow(twi_star.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap='viridis',
                         norm = colors.LogNorm(vmin=0.1, vmax=2e7)
                         )
    axs[m,n].set_title(r"$\gamma=%.2f$, $\lambda=%.2f$"%(df_params['gam'][i], df_params['lam'][i]), fontsize=9)
    axs[m,n].text(0.05, 
                0.95, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                ) 

    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=None, wspace=0.15, hspace=0.15)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) #Left, Bottom, Width, Height
fig.colorbar(im, cax=cbar_ax, label=r"$a'/(|\nabla' z'| \cos^2 \theta)$")
axs[-1, 0].set_ylabel(r'$y/\ell_g$')
axs[-1, 0].set_xlabel(r'$x/\ell_g$')
plt.savefig('%s/TI_%s.png'%(save_directory, base_output_path), dpi=300)


#%% DupuitLEM: aggregate view slope area

nrows=3
ncols=2
plot_runs = np.array([0,2,5,24,26,29])
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
 
    S = grid.at_node['slope_D8']
    A = grid.at_node['drainage_area']
    qstar = grid.at_node['qstar']  

    # make dimensionless versions
    a_star = (A/grid.dx)/lg
    S_star = S*lg/hg

    sc = axs[m,n].scatter(a_star[grid.core_nodes], 
                  S_star[grid.core_nodes], 
                  s=8, 
                  alpha=0.5, 
                  c=qstar[grid.core_nodes], 
                  cmap='plasma_r',
                  vmin=0.0,
                  vmax=1.0,
                  )
    axs[m,n].text(0.86, 
                0.95, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                ) 
    axs[m,n].set_xscale('log')
    axs[m,n].set_yscale('log')
    axs[m,n].set_title(r"$\gamma=%.2f$, $\lambda=%.2f$"%(df_params['gam'][i], df_params['lam'][i]), fontsize=9)

axs[-1,0].set_xlabel(r"$a'$")
axs[-1,0].set_ylabel(r"$|\nabla'z'|$")
fig.subplots_adjust(right=0.75, hspace=0.3, wspace=0.3)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, label=r'$Q^*$', orientation="vertical")
cbar.solids.set_edgecolor("face")
plt.savefig('%s/slope_area_qstar_%s.png'%(save_directory, base_output_path), dpi=300)


#%% DupuitLEM: aggregate view steepness curvature

nrows=3
ncols=2
plot_runs = np.array([0,2,5,24,26,29])
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
 
    S = grid.at_node['slope_D8']
    A = grid.at_node['drainage_area']
    curvature = grid.at_node['curvature']
    qstar = grid.at_node['qstar']  

    # make dimensionless versions
    a_star = (A/grid.dx)/lg
    S_star = S*lg/hg
    steepness_star = np.sqrt(a_star)*S_star
    curvature_star = curvature*lg**2/hg
    
    x = np.linspace(0,max(steepness_star), 10)
    y_theory = x - 1
    sc = axs[m,n].scatter(steepness_star[grid.core_nodes], 
                  curvature_star[grid.core_nodes], 
                  s=8, 
                  alpha=0.5, 
                  c=qstar[grid.core_nodes], 
                  cmap='plasma_r',
                  vmin=0.0,
                  vmax=1.0,
                  )
    axs[m,n].text(0.05, 
                0.95, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                ) 
    axs[m,n].plot(x,y_theory, 'k--', label=r'$\sqrt{a^*} |\nabla^* z^*| - 1$')
    axs[m,n].set_title(r"$\gamma=%.2f$, $\lambda=%.2f$"%(df_params['gam'][i], df_params['lam'][i]), fontsize=9)

axs[-1,0].set_xlabel(r"$\sqrt{a'} |\nabla' z'|$")
axs[-1,0].set_ylabel(r"$\nabla'^{2} z'$")
fig.subplots_adjust(right=0.75, hspace=0.3)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, label=r'$Q^*$', orientation="vertical")
cbar.solids.set_edgecolor("face")
axs[0,0].legend(frameon=False)
plt.savefig('%s/curv_steep_qstar_%s.png'%(save_directory, base_output_path), dpi=300)

#%% DupuitLEM: aggregate view TI curvature

nrows=3
ncols=2
plot_runs = np.array([0,2,5,24,26,29])
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    curvature = grid.at_node['curvature']
    S4 = grid.at_node['slope_D4']
    twi = grid.at_node['topographic__index_D4']/np.cos(np.arctan(S4))**2
    qstar = grid.at_node['qstar']
     
    # make dimensionless versions
    twi_star = twi*hg/lg**2
    curvature_star = curvature*lg**2/hg
    
    sc = axs[m,n].scatter(twi_star[grid.core_nodes], 
                  curvature_star[grid.core_nodes], 
                  s=8, 
                  alpha=0.25, 
                  c=qstar[grid.core_nodes], 
                  cmap='plasma_r',
                  vmin=0.0,
                  vmax=1.0,
                  )
    axs[m,n].text(0.05, 
                0.95, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                ) 
    axs[m,n].set_xscale('log')
    axs[m,n].set_title(r"$\gamma=%.2f$, $\lambda=%.2f$"%(df_params['gam'][i], df_params['lam'][i]), fontsize=9)

axs[-1,0].set_xlabel(r"$a'/(|\nabla' z'| \cos^2 \theta)$")
axs[-1,0].set_ylabel(r"$\nabla'^{2} z'$")
fig.subplots_adjust(right=0.75, hspace=0.3)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, label=r'$Q^*$', orientation="vertical")
cbar.solids.set_edgecolor("face")
# plt.tight_layout()
plt.savefig('%s/curv_twi_qstar_%s.png'%(save_directory, base_output_path), dpi=300)

#%% DupuitLEM: geomorphic balance

plot_runs=model_runs
nrows=6
ncols=5
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,6))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    D = df_params['D'][i]
    K = df_params['K'][i]
    U = df_params['U'][i]
    v0 = df_params['v0'][i]
    
    S = grid.at_node['slope_D8']
    A = grid.at_node['drainage_area']
    dx_star = grid.dx/lg
    steepness = np.sqrt(A/grid.dx)*S
    curvature = grid.at_node['curvature']
    qstar = grid.at_node['qstar']
    
    t1 = D/(K*np.sqrt(v0))*curvature[grid.core_nodes]/steepness[grid.core_nodes]
    t2 = U/(K*np.sqrt(v0))*1/steepness[grid.core_nodes]
    
    axs[m, n].scatter(t1+t2, qstar[grid.core_nodes], s=10, alpha=0.2)
    axs[m,n].plot([0,1.0],[0.0,1.0], 'k--', label='1:1')
    axs[m, n].text(0.05, 
                    0.95, 
                    str(i), 
                    transform=axs[m, n].transAxes, 
                    fontsize=10, 
                    verticalalignment='top',
                    )
    axs[m, n].ticklabel_format(style='sci')
    axs[m, n].tick_params(axis='both', which='major', labelsize=6)
    # axs[m, n].axis('equal')
    axs[m, n].set_xlim((-0.1, 1.1))
    axs[m, n].set_ylim((-0.1, 1.1))
    if m%nrows != nrows-1:
        axs[m, n].set_xticklabels([])
    if n%ncols != 0:
        axs[m, n].set_yticklabels([])
    
    
plt.subplots_adjust(wspace=0.35, hspace=0.2)
axs[-1, 0].legend(frameon=False, loc=4, fontsize=8)
axs[-1, 0].set_xlabel('RHS')
axs[-1, 0].set_ylabel('$Q^*$')
fig.suptitle('Geomorphic Balance', fontsize=14, y=0.92)
plt.savefig('%s/geomorphic_balance_%s.png'%(save_directory, base_output_path), dpi=300)

#%% DupuitLEM: hydrologic balance

plot_runs=model_runs
nrows=6
ncols=5
plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)


fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,6))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    ks = df_params['ksat'][i]
    b = df_params['b'][i]
    p = df_params['p'][i]
    
    dx = max(grid.length_of_face)
    
    elev = grid.at_node['topographic__elevation']
    h = grid.at_node['water_table__elevation'] - grid.at_node['aquifer_base__elevation']
    S = grid.at_node['slope_D8']
    S4 = grid.at_node['slope_D4']
    S8 = grid.at_node['slope_D8']
    twi = grid.at_node['topographic__index_D4']/np.cos(np.arctan(S4))**2
    
    # S6 = grid.at_node['slope_D6']
    # twi = grid.at_node['topographic__index_D6']/np.cos(np.arctan(S6))**2
    A = grid.at_node['drainage_area']
    a = A/dx
    dx_star = dx/lg
    qstar = grid.at_node['qstar']
    
    # make dimensionless versions
    twi_star = twi*hg/lg**2
    
    inds = np.logical_and(qstar>0.001, A>2*dx**2)
    # inds = qstar>0.001
    t2 = (ks*b)/p * 1/twi
    # t2_alt = df_params['gam'][i]*(1+df_params['lam'][i])/df_params['lam'][i] * 1/twi_star[inds]
    # t2_alt1 = (ks*h[inds])/(p*a[inds])*(dhdx_D4[inds]+S4[inds])*cosa[inds]**2
    
    Qgw_in = grid.at_node['gw_flux_in']
    Qgw_out = grid.at_node['gw_flux_out']
    Qgw_in_max = grid.at_node['gw_flux_in_max']
    Qgw_out_max = grid.at_node['gw_flux_out_max']

    t2_alt3 = Qgw_in_max/(p*A)
    
    axs[m, n].scatter(1-t2[inds], qstar[inds], s=6, alpha=0.15)#, c=steepness_star[inds], cmap='viridis')
    axs[m,n].plot([0,1.0],[0.0,1.0], 'k--', label='1:1')
    axs[m, n].text(0.05, 
                    0.95, 
                    str(i), 
                    transform=axs[m, n].transAxes, 
                    fontsize=10, 
                    verticalalignment='top',
                    )
    axs[m, n].ticklabel_format(style='sci')
    axs[m, n].tick_params(axis='both', which='major', labelsize=6)
    axs[m, n].set_xlim((-1.01, 1.1))
    axs[m, n].set_ylim((-0.1, 1.1))
    # axs[m, n].axis('equal')
    if m%nrows != nrows-1:
        axs[m, n].set_xticklabels([])
    if n%ncols != 0:
        axs[m, n].set_yticklabels([])
    
plt.subplots_adjust(wspace=0.35, hspace=0.2)
fig.suptitle('Hydrologic Balance', fontsize=14, y=0.92)
axs[-1, 0].set_xlabel('RHS')
axs[-1, 0].set_ylabel('$Q^*$')
axs[-1, 0].legend(frameon=False, loc=4, fontsize=8)
plt.savefig('%s/hydrologic_balance_%s.png'%(save_directory, base_output_path), dpi=300)

#%% DupuitLEM: hydromorphic balance 3D manifold

i = 24

manifold_curv= lambda steep, ti, gam: steep - gam*(steep/ti) - 1

grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
lg = df_params['lg'][i]
hg = df_params['hg'][i]
S = grid.at_node['slope_D8']
A = grid.at_node['drainage_area']
curvature = grid.at_node['curvature']
S4 = grid.at_node['slope_D4']
twi = grid.at_node['topographic__index_D4']/np.cos(np.arctan(S4))**2
qstar = grid.at_node['qstar']

# make dimensionless versions
a_star = (A/grid.dx)/lg
S_star = S*lg/hg
steepness_star = np.sqrt(a_star)*S_star
curvature_star = curvature*lg**2/hg
twi_star = twi*hg/lg**2

steep = np.linspace(0.1, 5, 100)
ti = np.geomspace(1.5,1e7, 100)
# steep = np.linspace(1.0, 75, 100)
# ti = np.geomspace(5,1e7, 100)
Steep, Ti = np.meshgrid(steep, ti)
Curv = manifold_curv(Steep, Ti, df_params['gam'][i])

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Steep, np.log(Ti), Curv, alpha=0.6, color='0.3')
sc = ax.scatter(steepness_star[grid.core_nodes], 
                 np.log(twi_star[grid.core_nodes]), 
                 curvature_star[grid.core_nodes], 
                  s=3, 
                  alpha=0.8, 
                  c=qstar[grid.core_nodes], 
                  cmap='plasma_r',
                  vmin=0.0,
                  vmax=1.0,
                  )
fig.colorbar(sc, label=r'$Q^*$')  
ax.set_xlabel(r"$S_z$")
ax.set_ylabel(r"log($T_z$)")
ax.set_zlabel(r"$C_z$")
# ax.set_zlim((-2, 75))
ax.set_zlim((-2, 5))
ax.view_init(elev=30., azim=230)
plt.tight_layout()
plt.savefig('%s/manifold_%s_%d.png'%(save_directory, base_output_path, i), dpi=300)

#%% DupuitLEM: hydromorphic balance map view

xmin =0; xmax=87.5; ymin=0; ymax=87.5

plot_runs = np.array([29,5,24,0])

cmap1 = copy.copy(cm.hot)
cmap1.set_bad(color='gray')
cmap1.set_over(color='magenta')
cmap1.set_under(color='blue')
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
for j in range(len(plot_runs)):
    i = plot_runs[j]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    S = grid.at_node['slope_D8']
    A = grid.at_node['drainage_area']
    curvature = grid.at_node['curvature']
    S4 = grid.at_node['slope_D4']
    twi = grid.at_node['topographic__index_D4']/np.cos(np.arctan(S4))**2
    qstar = grid.at_node['qstar']
    
    # make dimensionless versions
    a_star = (A/grid.dx)/lg
    S_star = S*lg/hg
    steepness_star = np.sqrt(a_star)*S_star
    curvature_star = curvature*lg**2/hg
    twi_star = twi*hg/lg**2
    
    inds = curvature_star<0
    T1 = df_params['gam'][i]*1/twi_star
    T1[grid.boundary_nodes] = np.nan
    T1[inds] = np.nan
    
    T2 = curvature_star/steepness_star
    T2[grid.boundary_nodes] = np.nan
    T2[inds] = np.nan
    
    T3 = 1/steepness_star
    T3[grid.boundary_nodes] = np.nan
    T3[inds] = np.nan
    
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5

    im = axs[j, 0].imshow(T1.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                          cmap=cmap1,
                         vmin=0.0,
                         vmax=1.0,
                         interpolation='none'
                         )
    axs[j,0].set_ylabel(r'\LARGE (%d)'%i, rotation=0, labelpad=20)
    
    im = axs[j, 1].imshow(T2.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap=cmap1,
                         vmin=0.0,
                         vmax=1.0,
                         )

    im = axs[j, 2].imshow(T3.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap=cmap1,
                         vmin=0.0,
                         vmax=1.0,
                         )

    im = axs[j, 3].imshow((T1+T2+T3-1).reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap=cmap1,
                         vmin=0.0,
                         vmax=1.0,
                         )

    if j==0:
        axs[j, 0].set_title(r"$\frac{\gamma}{T_z}$", fontsize=20, y=1.08)
        axs[j, 1].set_title(r"$\frac{C_z}{S_z}$", fontsize=20, y=1.08)
        axs[j, 2].set_title(r"$\frac{1}{S_z}$", fontsize=20, y=1.08)
        axs[j, 3].set_title(r'$\frac{\gamma}{T_z}+\frac{C_z}{S_z}+\frac{1}{S_z}-1$', fontsize=13, y=1.08)


    for k in range(0,4):
        axs[j, k].set_xlim((xmin,xmax))
        axs[j, k].set_ylim((ymin,ymax))
        
        if j != 3:
            axs[j, k].set_xticklabels([])
        else:
            axs[j, k].set_xticks([0, 25, 50, 75])
        if k != 0:
            axs[j, k].set_yticklabels([])
        else:
            axs[j, k].set_yticks([0, 25, 50, 75])


plt.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6]) #Left, Bottom, Width, Height
fig.colorbar(im, cax=cbar_ax, label="Term value", extend='both')
axs[-1, 0].set_xlabel(r'$x/\ell_g$')
plt.savefig('%s/hydromorphic_balance_%s.png'%(save_directory, base_output_path), dpi=400)

#%% DupuitLEM: load data for hillslope length

# specify model runs
base_output_paths = ['steady_sp_3_17', 'steady_sp_3_16', 'steady_sp_3_15']  
model_runs = np.arange(40)

# load params
dfs = []
for base_output_path in base_output_paths:
    dfs.append(pickle.load(open('%s/%s/parameters.p'%(directory,base_output_path), 'rb')))
df_params = pd.concat(dfs, axis=0, ignore_index=True)
    

# load results
dfs_tog = []
for base_output_path in base_output_paths:
    
    dfs = []
    for ID in model_runs:
        
        d = pickle.load(open('%s/%s/output_ID_%d.p'%(directory,base_output_path, ID), 'rb'))
        df = pd.DataFrame([d])
        dfs.append(df)
    dfs_tog.append(pd.concat(dfs, axis=0, ignore_index=True))
df_results = pd.concat(dfs_tog, axis=0, ignore_index=True)

#%% DupuitLEM: hillslope length for all model results

lam_all = pd.unique(df_params['lam'])
gam_all = pd.unique(df_params['gam'])
alpha_all = pd.unique(df_params['alpha'])
lg = df_params['lg'][0]

viridis = cm.get_cmap('viridis', len(lam_all))
col = viridis.colors
n = len(gam_all)
m = 40
linestyles = [':', '--', '-' ]

# plot shaded scaling lines
fig, ax = plt.subplots(figsize=(5,4))
y = np.geomspace(1,10,5)
for j in range(len(y)-1):
    ax.plot(np.linspace(0.9,12,50), y[j]*np.linspace(0.9,12,50)**(2/3), 'k-', alpha=0.3, linewidth=1.0)
ax.plot(np.linspace(0.9,12,50), y[-1]*np.linspace(0.9,12,50)**(2/3), 'k-', alpha=0.3, linewidth=1.0, label=r'$c*\gamma^{2/3}$')

# plot data
for k in range(len(alpha_all)):
    for i in range(len(lam_all)):
        if k == 2:
            ax.plot(gam_all, df_results['mean hillslope len'][n*i+k*m:n*i+n+k*m]/lg, label=r'$\lambda$=%.2f'%lam_all[i], color=col[i], linestyle=linestyles[k])
        else: 
            ax.plot(gam_all, df_results['mean hillslope len'][n*i+k*m:n*i+n+k*m]/lg, color=col[i], linestyle=linestyles[k])
ax.set_xlabel(r'$\gamma$ [-]')
ax.set_ylabel(r'$L_h/\ell_g$ [-]')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim((1, 50))
ax.set_xlim((0.9, 12))
legend0 = ax.legend(facecolor='white', framealpha=1.0)
frame = legend0.get_frame()
frame.set_linewidth(0)

# dummy lines to make a legend of different linestyles, independent of lambda legend
dummy_lines = []
for style in linestyles:
    dummy_lines.append(ax.plot([],[], c="black", ls = style)[0])
legend1 = plt.legend([dummy_lines[i] for i in [1,2,0]], [r'$\alpha=%.3f$'%alpha_all[1], r'$\alpha=%.3f$'%alpha_all[2], r'$\alpha=%.3f$'%alpha_all[0]], loc=(0.65,0.02), frameon=False) 
ax.add_artist(legend0)


plt.tight_layout()
plt.savefig('%s/length_scales_%s.png'%(save_directory, base_output_path), dpi=300)

#%% steady state relief condition

lam_all = pd.unique(df_params['lam'])
gam_all = pd.unique(df_params['gam'])

viridis = cm.get_cmap('cividis', 5)
colors = viridis.colors

hg = df_params['hg'][0] # all the same hg and lg
lg = df_params['lg'][0]

linestyles = [':', '-.', '--', '-', (0, (3, 1, 1, 1, 1, 1))]

fig, ax = plt.subplots(nrows=2, figsize=(5,8))
for i in range(len(df_results)):
    
    lam_index = np.where(lam_all == df_params['lam'][i])[0][0]
    gam_index = np.where(gam_all == df_params['gam'][i])[0][0]
    df_relief = pd.read_csv('%s/%s/relief_change_%d.csv'%(directory, base_output_path, i))

    ax[0].plot(df_relief['t_nd'], df_relief['r_nd'], color=colors[gam_index], linestyle=linestyles[lam_index])    
    ax[1].plot(df_relief['t_nd'], abs(df_relief['drdt_nd']), color=colors[gam_index], linestyle=linestyles[lam_index])

ax[1].set_xlabel(r'$t/t_g$ [-]')
ax[0].set_ylabel(r'$R_h/h_g$ [-]')
ax[1].set_ylabel(r'$|\frac{d R_h/h_g}{dt}|$ [-]')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
plt.tight_layout()
plt.savefig('%s/steady_state_cond_%s.png'%(save_directory, base_output_path), dpi=300)
