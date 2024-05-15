# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:54:31 2022

@author: dgbli
"""
#%%
import glob
import numpy as np
import pandas as pd
import copy
import linecache
import statsmodels.api as sm 

from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource
from landlab.io.netcdf import from_netcdf
plt.rc('text', usetex=True)

from generate_colormap import get_continuous_cmap


# directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
directory = '/Users/dlitwin/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'stoch_gam_sigma_15' #'stoch_ai_sigma_10' #
model_runs = np.arange(25)
nrows = 5
ncols = 5

#%% load results and parameters

dfs = []
for ID in model_runs:
    try:
        df = pd.read_csv('%s/%s/output_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    except FileNotFoundError:
        df =  pd.DataFrame(columns=df.columns)
    dfs.append(df)
df_results = pd.concat(dfs, axis=1, ignore_index=True).T
df_results.to_csv('%s/%s/results.csv'%(directory,base_output_path), index=True, float_format='%.3e')
       
dfs = []
for ID in model_runs:
    df = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    dfs.append(df)
df_params = pd.concat(dfs, axis=1, ignore_index=True).T
df_params.to_csv('%s/%s/params.csv'%(directory,base_output_path), index=True, float_format='%.3e')

# for most runs, plot all 
set_name = 'all'
plot_runs = model_runs

# plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)
plot_array = np.flipud(plot_runs.reshape((nrows, ncols))) # note flipped!

df_params['lambda'] = (125*df_params['v0']/df_params['lg'])
CI = (df_params['Nx']*df_params['v0']/df_params['lg'])**(3/2)
#%% steady state  - sigma ai

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

lines = [(0, (1, 1)), # dotted
         (0, (5, 5)), # dashed
         (0, (5, 1)), # densely dashed
         (0, (3, 5, 1, 5)), #dashdotted
         (0, (3, 1, 1, 1)), #densely dashdotted
    ]

ai_all = np.unique(df_params['ai'])
sigma_all = np.unique(df_params['sigma'])

viridis = cm.get_cmap('viridis', len(ai_all))
cmap1 = viridis.colors


# runs = [0,4,12,20,24]
plt.figure(figsize=(6,5))
for ID in model_runs:
    
    ai = df_params['ai'][ID]
    sigma = df_params['sigma'][ID]
    
    ai_ID = np.where(ai_all==ai)[0][0]
    sig_ID = np.where(sigma_all==sigma)[0][0]
    
    # df_z_change = pd.read_csv('%s/%s/z_change_%d.csv'%(directory, base_output_path,ID))
    df_r_change = pd.read_csv('%s/%s/relief_change_%d.csv'%(directory, base_output_path,ID))    
    plt.plot(df_r_change['t_nd'][1:], df_r_change['r_nd'][1:] - df_r_change['r_nd'][1], color=cmap1[ai_ID], linestyle=lines[sig_ID])
    
for i, sig in enumerate(sigma_all):
    plt.plot([-100], [-100], linestyle=lines[i], color='k', label='$\sigma=%.1f$'%sig)
for i, ai in enumerate(ai_all):
    plt.plot([-100], [-100], color=cmap1[i], label='Ai=%.1f'%ai)

plt.xlim((-50,2050))
plt.ylim((-50,1250))
plt.legend(frameon=False)
plt.xlabel(r'$t/t_g$', fontsize=14)
plt.ylabel(r'$\bar{z} / h_g$', fontsize=14)
plt.savefig('%s/%s/r_change.pdf'%(directory, base_output_path), transparent=True, dpi=300)

#%% steady state  - gamma sigma

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

lines = [(0, (1, 1)), # dotted
         (0, (5, 5)), # dashed
         (0, (5, 1)), # densely dashed
         (0, (3, 5, 1, 5)), #dashdotted
         (0, (3, 1, 1, 1)), #densely dashdotted
    ]

gam_all = np.unique(df_params['gam'])
sigma_all = np.unique(df_params['sigma'])

viridis = cm.get_cmap('viridis', len(gam_all))
cmap1 = viridis.colors


# runs = [0,4,12,20,24]
plt.figure(figsize=(6,5))
for ID in model_runs:
    
    gam = df_params['gam'][ID]
    sigma = df_params['sigma'][ID]
    
    gam_ID = np.where(gam_all==gam)[0][0]
    sig_ID = np.where(sigma_all==sigma)[0][0]
    
    # df_z_change = pd.read_csv('%s/%s/z_change_%d.csv'%(directory, base_output_path,ID))
    df_r_change = pd.read_csv('%s/%s/relief_change_%d.csv'%(directory, base_output_path,ID))    
    plt.plot(df_r_change['t_nd'][1:], df_r_change['r_nd'][1:] - df_r_change['r_nd'][1], color=cmap1[gam_ID], linestyle=lines[sig_ID])
    
for i, sig in enumerate(sigma_all):
    plt.plot([-100], [-100], linestyle=lines[i], color='k', label='$\sigma=%.1f$'%sig)
for i, gam in enumerate(gam_all):
    plt.plot([-100], [-100], color=cmap1[i], label='$\gamma=%.1f$'%gam)

plt.xlim((-50,2050))
plt.ylim((-10,450))
plt.legend(frameon=False)
plt.xlabel(r'$t/t_g$', fontsize=14)
plt.ylabel(r'$\bar{z} / h_g$', fontsize=14)
plt.savefig('%s/%s/r_change.pdf'%(directory, base_output_path), transparent=True, dpi=300)

#%% relief quantiles

ID = 4
df_r_change = pd.read_csv('%s/%s/relief_change_%d.csv'%(directory, base_output_path,ID))
r = df_r_change['r_nd'].values - df_r_change['r_nd'][1]
rf = r[-1]
pr = np.array([0.1, 0.3, 0.6, 0.9])
rt = pr*rf
inds = np.array([np.argmin(abs(r-rn)) for rn in rt])
rs = r[inds]


dt = (df_params.dtg/df_params.tg*df_params['output_interval'])[ID]
# plt.plot(df_r_change['t_nd'][1:], df_r_change['r_nd'][1:])
plt.figure(figsize=(2, 1.5))
plt.plot(df_r_change['t_nd'][1:], df_r_change['r_nd'][1:] - df_r_change['r_nd'][1], 'k-')
plt.scatter(inds*dt, rs, color='r')
plt.xlabel(r't/$t_g$', fontsize=14)
plt.ylabel(r'$\bar{z}/h_g$', fontsize=14)
plt.tight_layout()
plt.savefig('%s/%s/r_change_%d.pdf'%(directory, base_output_path, ID), transparent=True, dpi=300)

#%% Budyko

# storage_diff = np.zeros(len(model_runs))
# for ID in model_runs:
#     df = pd.read_csv('%s/%s/q_s_dt_ID_%d.csv'%(directory, base_output_path, ID))
#     storage_diff[ID] = df['S'].iloc[-1] - df['S'].iloc[0]
# df_results['deltaS'] = storage_diff

# mass_bal = df_results['cum_recharge'] + df_results['cum_extraction'] - df_results['cum_runoff'] - df_results['cum_gw_export']

df_results['cum_et_calc'] = df_results['cum_precip'] - df_results['cum_runoff'] - df_results['cum_gw_export']

alt = (df_results['cum_precip']-df_results['cum_recharge'])/df_results['cum_precip']
# ai = (df_results['cum_pet']/df_results['cum_precip'])
ai = df_params['ai']

sigma = np.unique(df_params['sigma'])
viridis = cm.get_cmap('viridis', len(sigma))
vir = viridis.colors

size = 20
fig, axs = plt.subplots(ncols=3, figsize=(9,3))
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[0].plot(ai[cond], alt[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    axs[0].scatter(ai[cond], alt[cond], color=vir[j], alpha=1.0, s=size, zorder=100)
axs[0].plot([0,1], [0,1], 'k--')
axs[0].plot([1,2], [1,1], 'k--')
axs[0].set_xlabel('Ai')
# axs[0].set_xlabel(r'$\langle PET \rangle / \langle P \rangle$')
axs[0].set_ylabel(r'$\langle AET \rangle/\langle P \rangle$')

qe = df_results['qe_tot']/df_results['cum_precip']
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[1].plot(ai[cond], qe[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    sc = axs[1].scatter(ai[cond], qe[cond], color=vir[j], alpha=1.0, s=size, zorder=102, label=r'$\sigma$ = %.1f'%sig)
axs[1].set_xlabel('Ai')
axs[1].set_ylabel(r'$\langle Q_f \rangle /\langle P \rangle$')


qb = df_results['qb_tot']/((df_results['cum_precip']-df_results['cum_recharge']) + df_results['qb_tot'])
# qb = df_results['qb_tot']/((df_results['cum_precip']-df_results['qe_tot']))
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[2].plot(ai[cond], qb[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    sc = axs[2].scatter(ai[cond], qb[cond], color=vir[j], alpha=1.0, s=size, zorder=101, label=r'$\sigma$ = %.1f'%sig)
axs[2].set_xlabel('Ai')
axs[2].set_ylabel(r'$\langle Q_b \rangle /(\langle Q_b \rangle + \langle AET \rangle)$')


axs[2].legend(frameon=False)
fig.tight_layout()
plt.show()
# plt.savefig('%s/%s/budyko_et_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)


#%% baseflow index

ai = df_params['ai']
sigma = np.unique(df_params['sigma'])
bf = df_results['Qb/Q']
bf = bf.fillna(0.0)

viridis = cm.get_cmap('viridis', len(sigma))
vir = viridis.colors

size = 20
fig, ax = plt.subplots(figsize=(4,3))
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    ax.plot(ai[cond], bf[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    sc = ax.scatter(ai[cond], bf[cond], color=vir[j], alpha=1.0, s=size, zorder=102, label=r'$\sigma$ = %.1f'%sig)
ax.set_ylim((0,1))
ax.set_xlabel('Ai')
ax.set_ylabel(r'$\langle Q_b \rangle /\langle Q \rangle$')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('%s/%s/bfi_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)


#%% baseflow index - gamma

gam = df_params['gam']
sigma = np.unique(df_params['sigma'])
bf = df_results['Qb/Q']
bf = bf.fillna(0.0)

viridis = cm.get_cmap('viridis', len(sigma))
vir = viridis.colors

size = 20
fig, ax = plt.subplots(figsize=(4,3))
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    ax.plot(gam[cond], bf[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    sc = ax.scatter(gam[cond], bf[cond], color=vir[j], alpha=1.0, s=size, zorder=102, label=r'$\sigma$ = %.1f'%sig)
ax.set_ylim((0,1))
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\langle Q_b \rangle /\langle Q \rangle$')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('%s/%s/bfi_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)


#%%
plt.figure()
plt.scatter((df_results['cum_precip']-df_results['cum_recharge']) + df_results['qb_tot'], df_results['cum_precip']-df_results['qe_tot'])
plt.axline([0,0],slope=1)
# plt.xscale('log')
# plt.yscale('log')


#%% Budyko - gam sigma

df_results['cum_et_calc'] = df_results['cum_precip'] - df_results['cum_runoff'] - df_results['cum_gw_export']

alt = (df_results['cum_precip']-df_results['cum_recharge'])/df_results['cum_precip']
ai = (df_results['cum_pet']/df_results['cum_precip'])
gam = df_params['gam']

sigma = np.unique(df_params['sigma'])
viridis = cm.get_cmap('viridis', len(sigma))
vir = viridis.colors

size = 20
fig, axs = plt.subplots(ncols=3, figsize=(9,3))
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[0].plot(gam[cond], alt[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    axs[0].scatter(gam[cond], alt[cond], color=vir[j], alpha=1.0, s=size, zorder=100)
axs[0].axhline(y=max(ai), label=r'$\langle PET \rangle/\langle P \rangle$', linestyle='--', color='k')
# axs[0].plot([0,1], [0,1], 'k--')
# axs[0].plot([1,2], [1,1], 'k--')
axs[0].set_xlabel(r'$\gamma$')
axs[0].set_ylabel(r'$\langle AET \rangle/\langle P \rangle$')
axs[0].legend(frameon=False)

qe = df_results['qe_tot']/df_results['cum_precip']
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[1].plot(gam[cond], qe[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    sc = axs[1].scatter(gam[cond], qe[cond], color=vir[j], alpha=1.0, s=size, zorder=102, label=r'$\sigma$ = %.1f'%sig)
axs[1].set_xlabel(r'$\gamma$')
axs[1].set_ylabel(r'$\langle Q_f \rangle /\langle P \rangle$')


qb = df_results['qb_tot']/((df_results['cum_precip']-df_results['cum_recharge']) + df_results['qb_tot'])
# qb = df_results['qb_tot']/((df_results['cum_precip']-df_results['qe_tot']))
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[2].plot(gam[cond], qb[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    sc = axs[2].scatter(gam[cond], qb[cond], color=vir[j], alpha=1.0, s=size, zorder=101, label=r'$\sigma$ = %.1f'%sig)
axs[2].set_xlabel(r'$\gamma$')
axs[2].set_ylabel(r'$\langle Q_b \rangle /(\langle Q_b \rangle + \langle AET \rangle)$')


axs[2].legend(frameon=False)
fig.tight_layout()
plt.savefig('%s/%s/budyko_et_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)

#%% plot_runs hillshades

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,6)) #(8,6)
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
                        vert_exag=1, 
                        dx=grid.dx, 
                        dy=grid.dy), 
                    origin="lower", 
                    extent=(x[0], x[-1], y[0], y[-1]), 
                    cmap='gray',
                    )
    axs[m,n].text(0.04, 
                0.96, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=8, 
                verticalalignment='top',
                color='k',
                bbox=dict(ec='w',
                          fc='w', 
                          alpha=0.7,
                          boxstyle="Square, pad=0.1",
                          )
                )   
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])

# axs[-1, 0].set_ylabel(r'$y/\ell_g$')
# axs[-1, 0].set_xlabel(r'$x/\ell_g$')
plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=None, wspace=0.15, hspace=0.15)
plt.savefig('%s/%s/hillshade_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)

#%% individual cross sections

# inds = [0,5,16,28]
inds = [2, 30, 10]

for i in inds:

    fig, axs = plt.subplots(figsize=(4,1.5))
    
    hg = df_params['hg'][i] # all the same hg and lg
    lg = df_params['lg'][i]
    b = df_params['b'][i]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    base = grid.at_node['aquifer_base__elevation']
    wt_high = grid.at_node['wtrel_mean_end_storm']*b + base
    wt_low = grid.at_node['wtrel_mean_end_interstorm']*b + base
    middle_row = np.where(grid.x_of_node == np.median(grid.x_of_node))[0][1:-1]
    
    y = grid.y_of_node[middle_row]/lg
    axs.fill_between(y,elev[middle_row]/hg,base[middle_row]/hg,facecolor=(198/256,155/256,126/256) )
    axs.fill_between(y,wt_high[middle_row]/hg,base[middle_row]/hg,facecolor=(145/256,176/256,227/256), alpha=1.0) #
    # axs.fill_between(y,wt_low[middle_row]/hg,base[middle_row]/hg,facecolor='royalblue', alpha=1.0)
    axs.fill_between(y,base[middle_row]/hg,np.zeros_like(base[middle_row]),facecolor=(111/256,111/256,111/256))
    axs.set_xlim((min(y),max(y)))
    axs.set_ylim((0,np.nanmax(elev[middle_row]/hg)*1.05))
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('%s/%s/cross_section_%s_%d.png'%(directory, base_output_path, base_output_path,i), dpi=300, transparent=True)


#%% individual hillshades 

inds = [2, 30, 10]

for i in inds:
    
    fig, axs = plt.subplots(figsize=(4,5))
    
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    # grid = read_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    ls = LightSource(azdeg=135, altdeg=45)
    axs.imshow(
                ls.hillshade(elev.reshape(grid.shape).T, 
                    vert_exag=1, 
                    dx=grid.dx, 
                    dy=grid.dy), 
                origin="lower", 
                extent=(x[0], x[-1], y[0], y[-1]), 
                cmap='gray',
                ) 
    axs.axis('off')
    plt.savefig('%s/%s/hillshade_%s_%d.png'%(directory, base_output_path, base_output_path,i), dpi=300, transparent=True)

#%%

i = 0
fig, axs = plt.subplots(figsize=(4,1.5))

hg = df_params['hg'][i] # all the same hg and lg
lg = df_params['lg'][i]
b = df_params['b'][i]

grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt_high = grid.at_node['wtrel_mean_end_storm']*b + base
wt_low = grid.at_node['wtrel_mean_end_interstorm']*b + base
middle_row = np.where(grid.x_of_node == np.median(grid.x_of_node))[0][1:-1]

y = grid.y_of_node[middle_row]/lg
axs.fill_between(y,elev[middle_row]/hg,base[middle_row]/hg,facecolor=(198/256,155/256,126/256) )
axs.fill_between(y,wt_high[middle_row]/hg,base[middle_row]/hg,facecolor='cornflowerblue', alpha=1.0) #
axs.fill_between(y,wt_low[middle_row]/hg,base[middle_row]/hg,facecolor='darkslateblue', alpha=1.0)
axs.fill_between(y,base[middle_row]/hg,np.zeros_like(base[middle_row]),facecolor=(111/256,111/256,111/256))
axs.set_xlim((min(y),max(y)))
axs.set_ylim((0,np.nanmax(elev[middle_row]/hg)*1.05))
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
plt.tight_layout()

    
#%% recharge 

max_recharge = 0.0
max_extraction = 0.0

color_w = [i for i in model_runs if i%7 in [0,1,2]]
color_text = ['k' if i in color_w else 'w' for i in model_runs]


hex_colors = ['000000', '083D77', 'F95738', 'EE964B', 'F4D35E', 'EBEBD3'] # based on https://coolors.co/palette/083d77-ebebd3-f4d35e-ee964b-f95738
hex_decs = [0, 0.1, 0.6, 0.8, 0.9, 1.0 ]
cmap1 = get_continuous_cmap(hex_colors, float_list=hex_decs)

# cmap1 = copy.copy(cm.magma)
cmap1.set_bad('k')
cmap1.set_over('w')

var_recharge = np.zeros(len(plot_runs))
mean_recharge = np.zeros(len(plot_runs))
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,5)) #(8,5)
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    recharge = (grid.at_node['recharge_rate_mean_storm']*df_params['tr'][i])/(df_params['ds'][i])
    recharge[grid.boundary_nodes] = 9999 # to catch set_upper
    var_recharge[i] = np.std(recharge[grid.core_nodes])
    mean_recharge[i] = np.mean(recharge[grid.core_nodes])
    
    max_recharge = max(max_recharge, max(recharge))
    
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 

    im = axs[m, n].imshow(recharge.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                          cmap=cmap1,
                         # norm=colors.LogNorm(vmin=1e-3, vmax=1),
                         vmin=0.0,
                         vmax=1.0,
                         interpolation = 'none'
                         )
    axs[m,n].text(0.05, 
                0.95, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=8, 
                verticalalignment='top',
                color='w', #color_text[i],
                # bbox=dict(ec='w',
                #           fc='w', 
                #           alpha=0.7,
                #           boxstyle="Square, pad=0.1",
                #           )
                )
    axs[m, n].axis('off')

    
# plt.subplots_adjust(left=0.15, right=0.8, wspace=0.15, hspace=0.15)
plt.subplots_adjust(left=0.15, right=0.8, wspace=0.05, hspace=0.02)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) #Left, Bottom, Width, Height
fig.colorbar(im, cax=cbar_ax, label=r"$\langle r \rangle / p$") #, extend="min"
fig.patch.set_visible(False)
plt.savefig('%s/%s/recharge_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)


# fig, axs = plt.subplots( figsize=(3,3))
# for j, sig in enumerate(sigma):
#     cond = df_params['sigma'] == sig
#     axs.plot(df_params['ai'][cond], var_recharge[cond]/mean_recharge[cond], alpha=0.5, color=vir[j], linewidth=1.0)
#     axs.scatter(df_params['ai'][cond], var_recharge[cond]/mean_recharge[cond], color=vir[j], alpha=1.0, s=size, zorder=100)
# axs.set_xlabel('Ai')
# axs.set_ylabel(r'var')

# fig, axs = plt.subplots( figsize=(3,3))
# for j, sig in enumerate(sigma):
#     cond = df_params['sigma'] == sig
#     axs.plot(df_params['ai'][cond], mean_recharge[cond], alpha=0.5, color=vir[j], linewidth=1.0)
#     axs.scatter(df_params['ai'][cond], mean_recharge[cond], color=vir[j], alpha=1.0, s=size, zorder=100)
# axs.set_xlabel('Ai')
# axs.set_ylabel(r'var')

#%% individual recharge

i = 9

hex_colors = ['000000', '083D77', 'F95738', 'EE964B', 'F4D35E', 'EBEBD3'] # based on https://coolors.co/palette/083d77-ebebd3-f4d35e-ee964b-f95738
hex_decs = [0, 0.1, 0.6, 0.8, 0.9, 1.0 ]
cmap1 = get_continuous_cmap(hex_colors, float_list=hex_decs)
cmap1.set_bad('k')
cmap1.set_over('w')

fig, axs = plt.subplots(figsize=(4,4)) 
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
recharge = (grid.at_node['recharge_rate_mean_storm']*df_params['tr'][i])/(df_params['ds'][i])
recharge[grid.boundary_nodes] = 9999 # to catch set_upper

lg = df_params['lg'][i]
hg = df_params['hg'][i]
dx = grid.dx/df_params['lg'][i]
y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5

im = axs.imshow(recharge.reshape(grid.shape).T, 
                     origin="lower", 
                     extent=(x[0], x[-1], y[0], y[-1]), 
                      cmap=cmap1,
                     # norm=colors.LogNorm(vmin=1e-3, vmax=1),
                     vmin=0.0,
                     vmax=1.0,
                     interpolation = 'none'
                     )
axs.axis('off')
plt.savefig('%s/%s/recharge_%s_%d.pdf'%(directory, base_output_path, base_output_path, i), dpi=300, transparent=True)


#%% individual runoff

i = 16

fig, axs = plt.subplots(figsize=(4,4)) 
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
qstar = grid.at_node['surface_water_effective__discharge']/(df_params['p'][i]*grid.at_node['drainage_area'])

lg = df_params['lg'][i]
hg = df_params['hg'][i]
dx = grid.dx/df_params['lg'][i]
y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5

im = axs.imshow(qstar.reshape(grid.shape).T, 
                     origin="lower", 
                     extent=(x[0], x[-1], y[0], y[-1]), 
                      cmap='Blues',
                     # norm=colors.LogNorm(vmin=1e-3, vmax=1),
                     vmin=0.0,
                     vmax=1.0,
                     interpolation = 'none'
                     )
axs.axis('off')
plt.savefig('%s/%s/qstar_%s_%d.png'%(directory, base_output_path, base_output_path, i), dpi=300, transparent=True)



#%% Saturation class

# colorbar approach courtesy of https://stackoverflow.com/a/53361072/11627361, https://stackoverflow.com/a/60870122/11627361, 

# event
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,5)) #(8,5)
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    try:
        grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    except FileNotFoundError:
        pass
    
    sat_storm = grid.at_node['sat_mean_end_storm']
    sat_interstorm = grid.at_node['sat_mean_end_interstorm']
    sat_class = grid.at_node['saturation_class']
    labels = ["dry", "variable", "wet"]    
    
    # sat_class = grid.add_zeros('node', 'saturation_class')
    # sat_never = np.logical_and(sat_storm < 0.001, sat_interstorm < 0.001)
    # sat_always = np.logical_and(sat_interstorm > 0.999, sat_storm > 0.999)
    # sat_variable = ~np.logical_or(sat_never, sat_always)
    # sat_class[sat_never] = 0
    # sat_class[sat_variable] = 1
    # sat_class[sat_always] = 2

    
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    L1 = ["peru", "dodgerblue", "navy"]
    cmap = colors.ListedColormap(L1)
    norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)
    fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    im = axs[m, n].imshow(sat_class.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap=cmap,
                         norm=norm,
                         interpolation="none",
                         # alpha=0.5,
                         )
    axs[m,n].text(0.05, 
                0.95, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=8, 
                verticalalignment='top',
                color='w',
                # bbox=dict(ec='w',
                #           fc='w', 
                #           alpha=0.7,
                #           boxstyle="Square, pad=0.1",
                #           )
                )  
    # if m != nrows-1:
    #     axs[m, n].set_xticklabels([])
    # if n != 0:
    #     axs[m, n].set_yticklabels([])
    axs[m, n].axis('off')
    
plt.subplots_adjust(left=0.15, right=0.8, wspace=0.05, hspace=0.05)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) #Left, Bottom, Width, Height
fig.colorbar(im, cax=cbar_ax, format=fmt, ticks=np.arange(0,3))
# plt.savefig('%s/%s/sat_zones_%s.png'%(directory, base_output_path, base_output_path), dpi=300)
plt.savefig('%s/%s/sat_zones_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)


#%% individual sat class

inds = [10, 22]

for i in inds:
    
    fig, axs = plt.subplots(figsize=(4,5))
    
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    sat_storm = grid.at_node['sat_mean_end_storm']
    sat_interstorm = grid.at_node['sat_mean_end_interstorm']
    sat_class = grid.at_node['saturation_class']
    labels = ["dry", "variable", "wet"]    
 
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    L1 = ["peru", "dodgerblue", "navy"]
    cmap = colors.ListedColormap(L1)
    norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)
    fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    im = axs.imshow(sat_class.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap=cmap,
                         norm=norm,
                         interpolation="none",
                         )
    axs.axis('off')
    plt.savefig('%s/%s/sat_class_%s_%d.png'%(directory, base_output_path, base_output_path,i), dpi=300, transparent=True)


#%% sensitivity of saturation class: all

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,5)) #(6,5)
for ID in plot_runs:
    m = np.where(plot_array==ID)[0][0]
    n = np.where(plot_array==ID)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, ID))
    
    sat_end_storm = grid.at_node['sat_mean_end_storm']
    sat_end_interstorm = grid.at_node['sat_mean_end_interstorm']
    
    threshold = np.linspace(0.005, 0.10, 20)
    sat_never = np.zeros_like(threshold)
    sat_always = np.zeros_like(threshold)
    sat_variable = np.zeros_like(threshold)
    
    L1 = ["peru", "dodgerblue", "navy"]
    for i, thresh in enumerate(threshold):
        sat_n = np.logical_and(sat_end_storm < thresh, sat_end_interstorm < thresh)
        sat_a = np.logical_and(sat_end_interstorm > 1-thresh, sat_end_storm > 1-thresh)
        sat_v = ~np.logical_or(sat_n, sat_a)
        
        sat_never[i] = np.sum(sat_n[grid.core_nodes])/grid.number_of_core_nodes
        sat_always[i] = np.sum(sat_a[grid.core_nodes])/grid.number_of_core_nodes
        sat_variable[i] = np.sum(sat_v[grid.core_nodes])/grid.number_of_core_nodes

    axs[m,n].fill_between(threshold, sat_always, color=L1[2])
    axs[m,n].fill_between(threshold, sat_always, sat_variable+sat_always, color=L1[1])
    axs[m,n].fill_between(threshold, sat_variable+sat_always, np.ones_like(threshold), color=L1[0]) 
    axs[m,n].set_xlim((0,0.1))
    axs[m,n].set_ylim((0,1))
    axs[m,n].axvline(0.05, color='k', linestyle='--', linewidth=0.5)
    
    axs[m,n].text(0.8, 
                0.9, 
                str(ID), 
                transform=axs[m,n].transAxes, 
                fontsize=8, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w', alpha=0.5)
                )  
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    
        
plt.subplots_adjust(wspace=0.35, hspace=0.25)
plt.savefig('%s/%s/sat_sensitivity_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)

#%% baseflow separation

i=4
# import timeseries and a grid
df = pd.read_csv('%s/%s/q_s_dt_ID_%d.csv'%(directory, base_output_path, i))
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))

cores = grid.core_nodes
area = grid.cell_area_at_node

q_star = df['qs']/(df_params['p'][ID]*np.sum(area[cores]))
qb_star = df['qb']/(df_params['p'][ID]*np.sum(area[cores]))
S_star = df['S']/(df_params['b'][ID]*df_params['ne'][ID]*np.sum(area[cores]))

rec_inds = np.where(np.diff(df['qs'], prepend=0.0) < 0.0)[0]
q_star_rec = q_star[rec_inds]
S_star_rec = S_star[rec_inds]

t_star = np.cumsum(df['dt'])/df_params['td'][i]


fig, axs = plt.subplots(ncols=2, figsize=(8,3))
axs[0].plot(t_star,q_star, color='k', label="$Q^*$")
axs[0].plot(t_star, qb_star, color='b', label="$Q_b^*$")
axs[0].axvspan(272.2,272.8, color='g', alpha=0.2)
axs[0].set_yscale('log')
axs[0].set_xlabel('$t/t_d$', fontsize=14)
axs[0].set_ylabel('$Q^*$', fontsize=14)
axs[0].set_xlim((265,275))
axs[0].legend(frameon=False)

axs[1].plot(t_star,q_star, color='k')
axs[1].plot(t_star, qb_star, color='b')
axs[1].set_yscale('log')
axs[1].set_xlabel('$t/t_d$', fontsize=14)
axs[1].set_ylabel('$Q^*$', fontsize=14)
axs[1].set_xlim((272.2,272.6))

plt.tight_layout()
plt.savefig('%s/%s/baseflow_sep_%d.pdf'%(directory, base_output_path,i))

#%% saturation discharge 

cmap1 = copy.copy(cm.viridis)
cmap1.set_bad(cmap1(0))

wetting_frac = (df_results['cum_recharge'] - df_results['qe_tot'])/df_results['cum_precip']

i_max = 0

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,6)) #(10,6)

for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    # df = pickle.load(open('%s/%s/q_s_dt_ID_%d.p'%(directory, base_output_path, i), 'rb'))
    df = pd.read_csv('%s/%s/q_s_dt_ID_%d.csv'%(directory, base_output_path, i))
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))

    Atot = np.sum(grid.cell_area_at_node[grid.core_nodes])
    Qb = df['qb']/(Atot*df_params['p'][i])
    
    Q = df['qs_star']
    S = df['S_star']
    qs_cells = df['sat_nodes']/grid.number_of_cells #same as number of core nodes
    r = df['r']
    ibar = (df_params['ds'][i]/df_params['tr'][i])*np.sum(grid.cell_area_at_node)
    i_max = max(i_max, max(r/ibar))
    rstar = r/ibar
    
    sort = np.argsort(S)
    ind = 1000
    
    if i == 0:
        xp = np.geomspace(1e-4, 200, 10)
        axs[m, n].plot(xp, xp**2, color='gray', linestyle=(0,(5,5)), linewidth=0.5, label=r"$A_{sat}^* \sim Q_b^{*2}$")
        axs[m, n].plot(xp, 0.1*xp**(1/3), color='gray', linestyle='-', linewidth=1, label=r"$A_{sat}^* \sim Q_b^{*1/3}$")

    else:
        axs[m,n].plot(xp, xp**2, color='gray', linestyle=(0,(5,5)), linewidth=0.5)
        axs[m,n].plot(xp, 0.1*xp**(1/3), color='gray', linestyle='-', linewidth=1)

    sc = axs[m, n].scatter(Q[ind:], 
                            qs_cells[ind:], 
                            color='lightgray', 
                            s=4, 
                            alpha=0.05,
                            rasterized=True)
    
    sc = axs[m, n].scatter(Qb[ind:], #[sort[ind:]] 
                            qs_cells[ind:], 
                            c=S[ind:], 
                            s=4, 
                            alpha=0.2, 
                            # vmin=0.0,
                            # vmax=1.0,
                            norm=colors.LogNorm(vmin=1e-3, vmax=1), 
                            cmap=cmap1,
                            rasterized=True)

    axs[m, n].text(0.05, 
                    0.92, 
                    str(i), 
                    transform=axs[m, n].transAxes, 
                    fontsize=8, 
                    verticalalignment='top',
                    )
    axs[m, n].ticklabel_format(style='sci')
    axs[m, n].tick_params(axis='both', which='major')
    axs[m ,n].set_ylim((1e-3,1))
    axs[m ,n].set_xlim((1e-4,200))
    axs[m ,n].set_yscale('log')
    axs[m ,n].set_xscale('log')
    axs[m, n].set_yticks([1e-3, 1e-1])
    axs[m, n].set_xticks([1e-4, 1e0])
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    axs[m, n].minorticks_off()

fig.subplots_adjust(right=0.75, hspace=0.15, wspace=0.15)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, orientation="vertical", extend="min")
cbar.set_label(label='$S^*$', size=16)
cbar.solids.set_edgecolor("face")

axs[-1, 0].set_ylabel('$A^*_{sat}$', size=16)
# axs[-1, 0].set_xlabel('$Q^*$')
axs[-1, 0].set_xlabel('$Q_b^*$', size=16)
fig.legend(bbox_to_anchor=(0.85,0.1), facecolor='w', edgecolor='w', loc=8)


# plt.savefig('%s/%s/Q_sat_S_%s.png'%(directory, base_output_path, set_name), dpi=300)
plt.savefig('%s/%s/Qb_sat_S_%s.pdf'%(directory, base_output_path, set_name), dpi=300)
plt.close()
# 
#%% storage discharge 

cmap1 = copy.copy(cm.viridis)
cmap1.set_bad('r')

i_max = 0

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,6)) #(8,6)
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    # df = pickle.load(open('%s/%s/q_s_dt_ID_%d.p'%(directory, base_output_path, i), 'rb'))
    df = pd.read_csv('%s/%s/q_s_dt_ID_%d.csv'%(directory, base_output_path, i))
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    ps = np.sum(grid.at_node['saturation_class'] == 2)/grid.number_of_core_nodes
    psv = ps + np.sum(grid.at_node['saturation_class'] == 1)/grid.number_of_core_nodes
    
    Atot = np.sum(grid.cell_area_at_node[grid.core_nodes])
    Qb = df['qb']/(Atot*df_params['p'][i])
    
    r = df['r']
    Q = df['qs_star'].values[r == 0]
    S = df['S_star'].values[r == 0]
    qs_cells = (df['sat_nodes'].values/grid.number_of_cells)[r == 0] #same as number of core nodes

    ibar = (df_params['ds'][i]/df_params['tr'][i])*np.sum(grid.cell_area_at_node)
    rstar = r/ibar
    
    sort = np.argsort(qs_cells)
    
    # axs[m,n].grid(color='lightgray', linewidth=0.5)
    # axs[m,n].axhline(ps, color='navy', linestyle='--')
    # axs[m,n].axhline(psv, color='dodgerblue', linestyle='--')
    ind = 1000
    sc = axs[m, n].scatter(S[sort], 
                            Q[sort], 
                            c=qs_cells[sort], 
                            s=4, 
                            alpha=0.2, 
                            norm=colors.LogNorm(vmin=1e-3, vmax=1),
                            # vmin=0.0,
                            # vmax=1.0,
                            cmap='cividis',
                            rasterized=True)
    x1 = np.geomspace(1e-4, 200)
    if i==0:
        axs[m, n].plot(x1, x1**(2), color='teal', linewidth=1, linestyle='--', label=r'$S^{* 3/2}$' )
    else:
        axs[m, n].plot(x1, x1**(2), color='teal', linewidth=1, linestyle='--')
    # axs[m, n].axline((1,1), slope=1, color='gray', linestyle='--', linewidth=1, label='1:1')


    axs[m, n].text(0.05, 
                    0.95, 
                    str(i), 
                    transform=axs[m, n].transAxes, 
                    fontsize=10, 
                    verticalalignment='top',
                    )
    axs[m, n].ticklabel_format(style='sci')
    axs[m, n].tick_params(axis='both', which='major', labelsize=8)
    axs[m ,n].set_xlim((1e-3,1))
    axs[m ,n].set_ylim((1e-4,200))
    axs[m ,n].set_yscale('log')
    axs[m ,n].set_xscale('log')
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    
fig.subplots_adjust(right=0.75, hspace=0.15, wspace=0.15)
rect_cb = [0.775, 0.35, 0.03, 0.3]  #Left, Bottom, Width, Height
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, orientation="vertical", extend="min")
cbar.set_label(label='$A^*_{sat}$', size=16)
cbar.solids.set_edgecolor("face")

axs[-1, 0].set_xlabel('$S^*$', size=16)
axs[-1, 0].set_ylabel('$Q^*$', size=16)
fig.legend(bbox_to_anchor=(0.825,0.2), facecolor='w', edgecolor='w', loc=8)

# plt.savefig('%s/%s/Q_sat_S_%s.png'%(directory, base_output_path, set_name), dpi=300)
plt.savefig('%s/%s/Q_S_%s.pdf'%(directory, base_output_path, set_name), dpi=300)
plt.close()


#%% discussion sat areas

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

ID = 14

# load grid and full callback timeseries
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, ID))
df = pd.read_csv('%s/%s/q_s_dt_ID_%d.csv'%(directory, base_output_path, ID))
Atot = np.sum(grid.cell_area_at_node[grid.core_nodes])
Qb = df['qb']/(Atot*df_params['p'][ID])
Q = df['qs_star']
S = df['S_star']
qs_cells = df['sat_nodes']/grid.number_of_cells #same as number of core nodes

# load the callback timeseries associated with the short duration run
df1 = pd.read_csv('%s/%s/state_fluxes_%d.csv'%(directory, base_output_path, ID),header=None, names=['dt','r', 'qs', 'S', 'sat_nodes'])
qs_cells1 = df1['sat_nodes']/grid.number_of_cells #same as number of core nodes

# analysis on short duration run
df1['t'] = np.cumsum(df1['dt'])
is_r = df1['r'] > 0.0
pre_r = np.where(np.diff(is_r*1)>0.0)[0] #last data point before recharge
end_r = np.where(np.diff(is_r*1)<0.0)[0][1:] #last data point where there is recharge

qb = df1['qs'].copy()
q = df1['qs']
t = df1['t']
for i in range(len(end_r)):
    j = pre_r[i]
    k = end_r[i]
    slope = (q[k+1] - q[j])/(t[k+1]-t[j])
    qb[j+1:k+1] = q[j]+slope*(t[j+1:k+1] - t[j])
Qb1 = qb/(Atot*df_params['p'][ID])


# line-by-line finding mean sat state
sat_mean = []
for row in open('%s/%s/sat_%d.csv'%(directory, base_output_path, ID), 'r'):
    r = np.array(row.split(','),int)
    sat_mean.append(np.mean(r[grid.core_nodes]))
sat_mean = np.array(sat_mean)

# find closest values to mean sat state, save the whole sat array in 'lines'
vals = [0.00, 0.11, 0.15, 0.4]
# vals = [0.007, 0.05, 0.1, 0.5] # use for #4
vals_closest = []
idxs = []
lines = np.zeros((len(vals), len(r)))
for i, val in enumerate(vals):
    valc, idx = find_nearest(sat_mean, val)
    idxs.append(idx)
    vals_closest.append(valc)
    print(i, val, valc)
    
    line = linecache.getline('%s/%s/sat_%d.csv'%(directory, base_output_path, ID), idx)
    lines[i,:] = np.array(line.split(','),int)



#%% discussion - just sat colored hillshades

elev = grid.at_node['topographic__elevation']
dx = grid.dx/df_params['lg'][ID]
y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5

ls = LightSource(azdeg=135, altdeg=45)
cmap = colors.ListedColormap(['blue'])
ind = 0

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8,3))
for j in range(len(vals)):    
    
    axs[j].imshow(
        ls.hillshade(elev.reshape(grid.shape).T, 
            vert_exag=1.0, 
            dx=grid.dx, 
            dy=grid.dy), 
        origin="lower", 
        extent=(x[0], x[-1], y[0], y[-1]), 
        cmap='gray',
        alpha=0.5
        )
    sat = np.ma.masked_where(lines[j,:]==0, lines[j,:])
    axs[j].imshow(sat.reshape(grid.shape).T, 
            origin="lower", 
            extent=(x[0], x[-1], y[0], y[-1]),
            alpha=0.7,
            cmap=cmap,
            interpolation="none"
            )
    axs[j].set_xticks([0,100,200])
    axs[j].set_yticks([0,100,200])
    # ax.set_xlim((x[1],x[-2]))
    # ax.set_ylim((y[1],y[-2]))
    axs[j].set_title(r'$A^*_{sat}=$'+' ${0:2.0f}\%$'.format(vals_closest[j]*100))
    
    if j == 0:
        axs[j].set_xlabel(r'$x/\ell_g$')
        axs[j].set_ylabel(r'$y/\ell_g$')
    else:
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])

plt.savefig('%s/%s/sat_areas_detail_%d.pdf'%(directory, base_output_path, ID), dpi=300, transparent=True)

#%% discussion - single sat-discharge

fig, ax0 = plt.subplots(figsize=(5,4))
sc = ax0.scatter(Qb[ind:],
                        qs_cells[ind:], 
                        c=S[ind:], 
                        s=4, 
                        alpha=0.2, 
                        norm=colors.LogNorm(vmin=1e-3, vmax=1), 
                        cmap="viridis",
                        rasterized=True
                        )

for idx in idxs:
    ax0.scatter(Qb1[idx], qs_cells1[idx], color='k', alpha=1.0)
ax0.ticklabel_format(style='sci')
ax0.tick_params(axis='both', which='major')
# ax0.set_ylim((3e-3,1)) # for #4
# ax0.set_xlim((1e-4,10)) # for #4
ax0.set_ylim((5e-2,1))
ax0.set_xlim((5e-2,10))
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_ylabel('$A^*_{sat}$', size=16)
ax0.set_xlabel('$Q_b^*$', size=16)
cbar = fig.colorbar(sc, orientation="vertical", extend="min")
cbar.set_label(label='$S^*$', size=16)
cbar.solids.set_edgecolor("face")
plt.tight_layout()
    
plt.savefig('%s/%s/sat_discharge_%d.pdf'%(directory, base_output_path, ID), dpi=300, transparent=True)

#%% discussion - single sat-discharge dimensioned

IDs= [0, 4, 16, 28]
colors = ['tab:green', 'tab:red', 'tab:purple', 'tab:cyan']
fig, ax0 = plt.subplots(figsize=(9,5)) #(5,4.8)

for i, ID in enumerate(IDs):
    # load grid and full callback timeseries
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, ID))
    df = pd.read_csv('%s/%s/q_s_dt_ID_%d.csv'%(directory, base_output_path, ID))
    Atot = np.sum(grid.cell_area_at_node[grid.core_nodes])
    Qb = df['qb'] * 1000 *(1/(Atot*1e-6))
    S = df['S_star']
    qs_cells = df['sat_nodes']/grid.number_of_cells #same as number of core nodes
    
    ind = 0
    
    
    sc = ax0.scatter(Qb[ind:],
                            qs_cells[ind:], 
                            color=colors[i],
                            s=10, 
                            alpha=0.1, 
                            cmap="viridis",
                            rasterized=True
                            )
ax0.ticklabel_format(style='sci')
ax0.tick_params(axis='both', which='major')
ax0.set_ylim((1e-4,0.5))
ax0.set_xlim((0.01,200)) #(1,200)
ax0.set_yscale('log')
ax0.set_xscale('log')
# ax0.set_ylabel('$A^*_{sat}$', size=16)
# ax0.set_xlabel('$Q_b$ [L/s/km2]', size=16)
# cbar = fig.colorbar(sc, orientation="vertical", extend="min")
# cbar.set_label(label='$S^*$', size=16)
# cbar.solids.set_edgecolor("face")
plt.tight_layout()
    
plt.savefig('%s/%s/sat_discharge_alt.pdf'%(directory, base_output_path), dpi=300, transparent=True)

#%% Dunne discussion - gam sigma

mean_r_nd = np.zeros(len(plot_runs))
for i in plot_runs:
    
    try:
        grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
        elev = grid.at_node['topographic__elevation']
        r = elev - np.nanmin(elev)
        hmean = np.mean(grid.at_node['wtrel_mean_end_interstorm'][grid.core_nodes])*df_params['b'][i]
        # rnd = r/df_params['hg'][i]
        rnd = r/hmean
        mean_r_nd[i] = np.mean(rnd[grid.core_nodes])
    except:
        mean_r_nd[i] = np.nan


gam_u = np.unique(df_params.gam)
plasma = cm.get_cmap('plasma', len(gam_u)+1)
plas = plasma.colors

fig, ax = plt.subplots(figsize=(6,4))
for i, gam in enumerate(gam_u):
    
    ids = np.where(df_params.gam==gam)[0]
    
    ax.plot(df_results['sat_variable'][ids], mean_r_nd[ids], linestyle='--', color=plas[i])
    sc = ax.scatter(df_results['sat_variable'][ids], 
                mean_r_nd[ids], 
                color=plas[i], 
                s=df_params['sigma'][ids],
                label=r'$\gamma=%.1f$'%gam)
    
ax.legend(frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Proportion variably saturated')
ax.set_ylabel('Mean relief / $h_a$')
ax.set_xlim((1e-3, 1.0))
# ax.set_ylim((3,50))
# plt.savefig('%s/%s/relief_var_sat.pdf'%(directory, base_output_path), dpi=300)

# # alternate with flipped sigma and gamma representation
# sigma_u = np.unique(df_params.sigma)
# plasma = cm.get_cmap('plasma', len(sigma_u)+1)
# plas = plasma.colors

# fig, ax = plt.subplots(figsize=(6,4))
# for i, sig in enumerate(sigma_u):
    
#     ids = np.where(df_params.sigma==sig)[0]
    
#     ax.plot(df_results['sat_variable'][ids], mean_r_nd[ids], linestyle='--', color=plas[i])
#     sc = ax.scatter(df_results['sat_variable'][ids], 
#                 mean_r_nd[ids], 
#                 color=plas[i], 
#                 s=df_params['gam'][ids]*4,
#                 label=r'$\sigma=%.1f$'%sig)
    
# ax.legend(frameon=False)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('Proportion variably saturated')
# ax.set_ylabel('Mean relief / $h_g$')
# # plt.savefig('%s/%s/relief_var_sat.pdf'%(directory, base_output_path), dpi=300)

#%% Dunne discussion - gam sigma multi-run    

       
def calc_mean_relief(grid):
    elev = grid.at_node['topographic__elevation']
    x = np.unique(grid.x_of_node)
    varline = np.zeros(len(x))
    
    for i, xi in enumerate(x):
        row = np.where(grid.x_of_node == xi)[0][1:-1]
        zrow = elev[row]
        varline[i] = np.var(zrow)
    
    return np.sqrt(np.nanmean(varline))

def calc_horiz_elev_change(grid):
    elev = grid.at_node['topographic__elevation']
    x = np.unique(grid.x_of_node)
    
    rowf = np.where(grid.x_of_node == x[1])[0][1:-1]
    rowl = np.where(grid.x_of_node == x[-2])[0][1:-1]
    return np.nanmin(elev[rowl]) - np.nanmin(elev[rowf])


#%% understanding relief plots

i = 12
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
elev = grid.at_node['topographic__elevation']
        
# elevation

lg = df_params['lg'][i]
hg = df_params['hg'][i]
dx = grid.dx/df_params['lg'][i]
y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
zplot = elev.reshape(grid.shape).T/hg
fig, ax = plt.subplots(figsize=(4,3))
im = ax.imshow(zplot, 
                origin="lower", 
                extent=(x[0], x[-1], y[0], y[-1]), 
                cmap='gist_earth',
                interpolation = 'none'
                )
fig.colorbar(im, label=r'$z/h_g$')


# mean elevations along each axis
zplot1 = zplot[1:-1,1:-1]
xp = np.unique(grid.x_of_node)[1:-1]/lg
yp = np.unique(grid.y_of_node)[1:-1]/lg
plt.figure(figsize=(4,3))
plt.plot(xp,np.mean(zplot1,axis=0), label='horizontals-mean')
plt.plot(yp,np.mean(zplot1,axis=1), label='verticals-mean')
plt.xlabel(r'$x/\ell_g$ or $y/\ell_g$')
plt.ylabel(r'$\bar{z}/h_g$')
plt.legend(frameon=False)
plt.tight_layout()


# variance along each axis
plt.figure(figsize=(4,3))
plt.plot(xp,np.std(zplot1,axis=0), label='horizontals-stdev')
plt.plot(yp,np.std(zplot1,axis=1), label='verticals-stdev')
plt.xlabel(r'$x/\ell_g$ or $y/\ell_g$')
plt.ylabel(r'$std(z)/h_g$')
plt.legend(frameon=False)
plt.tight_layout()



#%% hillslope number vs variably sat area


names = ['stoch_gam_sigma_14', 'stoch_gam_sigma_15', 'stoch_gam_sigma_16']
cmaps = ['Reds', 'Blues', 'Greens']
plot_runs = np.arange(25)

plt.rc('axes', labelsize=14) 

## case 0 (Relief/hg), case 1 (relief/ha), case 2 (relief/(ha beta))
case = 2
out_names = ['hg', 'ha', 'ha_beta']

fig, ax = plt.subplots(figsize=(6,4.2))
for k, base_output_path in enumerate(names):

    # results
    dfs = []
    for ID in plot_runs:
        try:
            df = pd.read_csv('%s/%s/output_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
        except FileNotFoundError:
            df =  pd.DataFrame(columns=df.columns)
        dfs.append(df)
    df_results = pd.concat(dfs, axis=1, ignore_index=True).T
    
    # parameters
    dfs = []
    for ID in plot_runs:
        df = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
        dfs.append(df)
    df_params = pd.concat(dfs, axis=1, ignore_index=True).T
    df_params.to_csv('%s/%s/params.csv'%(directory,base_output_path), index=True, float_format='%.3e')
    
            
    # topography
    mean_r_nd = np.zeros(len(plot_runs))
    for i in plot_runs:
        try:
            grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
            elev = grid.at_node['topographic__elevation']
            r = elev - np.nanmin(elev)
            hmean = np.mean(grid.at_node['wtrel_mean_end_interstorm'][grid.core_nodes])*df_params['b'][i]
            
            ## case 0 (Relief/hg)
            if case == 0:
                # rnd = r/df_params['hg'][i]
                # mean_r_nd[i] = np.mean(rnd[grid.core_nodes])
                
                mean_r_nd[i] = calc_mean_relief(grid)/df_params['hg'][i]
            
            ## case 1 (relief/ha)
            elif case == 1:
                # rnd = r/hmean
                # mean_r_nd[i] = np.mean(rnd[grid.core_nodes])
                mean_r_nd[i] = calc_mean_relief(grid)/hmean
                
            # case 2 (relief/(ha beta))
            elif case == 2:
                # rnd = r/hmean
                # mean_r_nd[i] = np.mean(rnd[grid.core_nodes])/df_params['hi'][i]
                mean_r_nd[i] = calc_mean_relief(grid)/hmean/df_params['hi'][i]
                
            else:
                print("choose case 1, 2, 3")
            
        except:
            mean_r_nd[i] = np.nan
    
    gam_u = np.unique(df_params.gam)
    # plasma = cm.get_cmap('plasma', len(gam_u)+1)
    # plas = plasma.colors
    
    col = cm.get_cmap(cmaps[k], len(gam_u)+1)
    cols = col(np.linspace(0.2,1.0,len(gam_u)+1))

    # plot
    for i, gam in enumerate(gam_u):
        
        ids = np.where(df_params.gam==gam)[0]
        
        # ax.plot(df_results['sat_variable'][ids], mean_r_nd[ids], linestyle='dotted', color=cols[i], linewidth=1.0)
        if base_output_path == names[0]:
            sc = ax.scatter(df_results['sat_variable'][ids], 
                        mean_r_nd[ids], 
                        color=cols[i], 
                        s=df_params['sigma'][ids],
                        # c=df_results['Qb/Q'][ids],
                        # vmin=0.0,
                        # vmax=1.0,
                        # cmap='plasma',
                        label=r'$\gamma=%.1f$'%gam)
        else:
            sc = ax.scatter(df_results['sat_variable'][ids], 
                        mean_r_nd[ids], 
                        color=cols[i], 
                        s=df_params['sigma'][ids],
                        # c=df_results['Qb/Q'][ids],
                        # vmin=0.0,
                        # vmax=1.0,
                        # cmap='plasma',
                        )            
# ax.legend(frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Proportion variably saturated')
if case == 0:
    ax.set_ylabel(r'$\overline{Z} / h_g$')
elif case == 1:
    ax.set_ylabel(r'$\overline{Z} / \langle h \rangle$')
else:
    ax.set_ylabel(r'$\left(\overline{Z} / \langle h \rangle \right) / \beta$')

fig.tight_layout()
plt.savefig('%s/%s/relief_alt_var_sat_%s.pdf'%(directory, base_output_path, out_names[case]), dpi=300)

# plt.rcParams.update(plt.rcParamsDefault)

#%% Dunne discussion - gam sigma multi-run        
 

names = ['stoch_gam_sigma_14', 'stoch_gam_sigma_15', 'stoch_gam_sigma_16']
cmaps = ['Reds', 'Blues', 'Greens']
plot_runs = np.arange(25)

plt.rc('axes', labelsize=14) 

fig, ax = plt.subplots(figsize=(6,4.2))
for k, base_output_path in enumerate(names):

    # results
    dfs = []
    for ID in plot_runs:
        try:
            df = pd.read_csv('%s/%s/output_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
        except FileNotFoundError:
            df =  pd.DataFrame(columns=df.columns)
        dfs.append(df)
    df_results = pd.concat(dfs, axis=1, ignore_index=True).T
    
    # parameters
    dfs = []
    for ID in plot_runs:
        df = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
        dfs.append(df)
    df_params = pd.concat(dfs, axis=1, ignore_index=True).T
    df_params.to_csv('%s/%s/params.csv'%(directory,base_output_path), index=True, float_format='%.3e')
    
            
    # topography
    mean_r_nd = np.zeros(len(plot_runs))
    mean_r_trend = np.zeros(len(plot_runs))
    for i in plot_runs:
        try:
            grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
            elev = grid.at_node['topographic__elevation']
            r = elev - np.nanmin(elev)
            hmean = np.mean(grid.at_node['wtrel_mean_end_interstorm'][grid.core_nodes])*df_params['b'][i]
            
            ## case 0 (Relief/hg)
            # rnd = r/df_params['hg'][i]
            # mean_r_nd[i] = np.mean(rnd[grid.core_nodes])
            mean_r_nd[i] = calc_mean_relief(grid)/df_params['hg'][i]
            mean_r_trend[i] = calc_horiz_elev_change(grid)/df_params['hg'][i]

        except:
            mean_r_nd[i] = np.nan
    
    gam_u = np.unique(df_params.gam)
    # plasma = cm.get_cmap('plasma', len(gam_u)+1)
    # plas = plasma.colors
    
    col = cm.get_cmap(cmaps[k], len(gam_u)+1)
    cols = col(np.linspace(0.2,1.0,len(gam_u)+1))

    # plot
    for i, gam in enumerate(gam_u):
        
        ids = np.where(df_params.gam==gam)[0]
        if base_output_path == names[0]:
            sc = ax.scatter(1-df_results['Qb/Q'][ids],
                        mean_r_nd[ids],
                        # color=cols[i], 
                        # s=df_params['sigma'][ids],
                        c=df_results['sat_variable'][ids],
                        vmin=0.0,
                        vmax=1.0,
                        cmap='plasma',
                        label=r'$\gamma=%.1f$'%gam)
        else:
            sc = ax.scatter(1-df_results['Qb/Q'][ids], 
                        mean_r_nd[ids],
                        # color=cols[i], 
                        # s=df_params['sigma'][ids],
                        c=df_results['sat_variable'][ids],
                        vmin=0.0,
                        vmax=1.0,
                        cmap='plasma',
                        )            
# ax.legend(frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlim((0.03,1))
# ax.set_ylim((2,200))
ax.set_xlabel(r'$ \langle Q_f \rangle / \langle Q \rangle$')
ax.set_ylabel(r'$\overline{Z} / h_g$')
fig.tight_layout()
# plt.savefig('%s/%s/relief_qfi_hg_sat.pdf'%(directory, base_output_path), dpi=300) #_sat


#%% dummy plots for legends

# sigma
fig, ax = plt.subplots(figsize=(6,4.2))
sizes = np.array([8, 16, 32, 64, 128])
for i, size in enumerate(sizes):
    sc = ax.scatter(-1, -1, s=size, label='%.1f'%size, color='k')
# fig.legend(*sc.legend_elements("sizes", num=5), loc='center')
ax.set_xlim((0,1))
ax.set_ylim((0,1))
fig.legend(frameon=False, title=r"$\sigma$", loc="center", title_fontsize=20)
ax.axis('off')
plt.savefig('%s/%s/leg_size.pdf'%(directory, base_output_path), dpi=300, transparent=True)


# gamma
col = cm.get_cmap('Greys', len(gam_u))
cols = col(np.linspace(0.2,1.0,len(gam_u)))

vals = [1.0, 2.0, 4.0, 8.0, 16.0]
fig, ax = plt.subplots(figsize=(6,4.2))
for i, color in enumerate(cols):
    sc = ax.scatter(-1, -1, color=color, label='%.1f'%vals[i])
# fig.legend(*sc.legend_elements("sizes", num=5), loc='center')
ax.set_xlim((0,1))
ax.set_ylim((0,1))
fig.legend(frameon=False, title=r"$\gamma$", loc="center", title_fontsize=20)
ax.axis('off')
plt.savefig('%s/%s/leg_col.pdf'%(directory, base_output_path), dpi=300, transparent=True)


# beta
fig, ax = plt.subplots(figsize=(6,4.2))
cols = ['b', 'r', 'g']
labels = [0.1, 0.5, 2.5]
for i, col in enumerate(cols):
    sc = ax.scatter(-1, -1, color=col, label='%.1f'%labels[i])
# fig.legend(*sc.legend_elements("sizes", num=5), loc='center')
ax.set_xlim((0,1))
ax.set_ylim((0,1))
fig.legend(frameon=False, title=r"$\beta$", loc="center", title_fontsize=20)
ax.axis('off')
plt.savefig('%s/%s/leg_beta.pdf'%(directory, base_output_path), dpi=300, transparent=True)

# colorbar
fig, ax = plt.subplots(figsize=(6,4.2))
sc = ax.scatter([-1,-1], [-1,-1], c=[0,1], cmap='plasma', vmin=0.0, vmax=1.0)
ax.set_xlim((0,1))
ax.set_ylim((0,1))
cbar = fig.colorbar(sc, orientation="vertical")
cbar.set_label(label='Proportion Variably Saturated', size=12)
cbar.solids.set_edgecolor("face")
plt.savefig('%s/%s/cbar_varsat.pdf'%(directory, base_output_path), dpi=300, transparent=True)


#%%

ID = 4
plot_runs = [120, 360, 760, 1320]

plt.rc('axes', labelsize=14) 

## case 0 (Relief/hg), case 1 (relief/ha), case 2 (relief/(ha beta))

fig, ax = plt.subplots(figsize=(6,4.2))

# results
dfs = []
for IT in plot_runs:
    try:
        df = pd.read_csv('%s/%s/output_ID_%d_%d.csv'%(directory,base_output_path, ID, IT), index_col=0)
    except FileNotFoundError:
        df =  pd.DataFrame(columns=df.columns)
    dfs.append(df)
df_results = pd.concat(dfs, axis=1, ignore_index=True).T

df4 = pd.read_csv('%s/%s/output_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, ID))
elev = grid.at_node['topographic__elevation']
r = elev - np.nanmin(elev)
hmean = np.mean(grid.at_node['wtrel_mean_end_interstorm'][grid.core_nodes])*df_params['b'][ID]
mean_r_nd4 = calc_mean_relief(grid)/df_params['hg'][ID]



# topography
mean_r_nd = np.zeros(len(plot_runs))
mean_r_trend = np.zeros(len(plot_runs))
for i, IT in enumerate(plot_runs):
  
    grid = from_netcdf('%s/%s/grid_%d_%d.nc'%(directory, base_output_path, ID, IT))
    elev = grid.at_node['topographic__elevation']
    r = elev - np.nanmin(elev)
    hmean = np.mean(grid.at_node['wtrel_mean_end_interstorm'][grid.core_nodes])*df_params['b'][ID]
    
    ## case 0 (Relief/hg)
    # rnd = r/df_params['hg'][i]
    # mean_r_nd[i] = np.mean(rnd[grid.core_nodes])
    mean_r_nd[i] = calc_mean_relief(grid)/df_params['hg'][ID]
    mean_r_trend[i] = calc_horiz_elev_change(grid)/df_params['hg'][ID]

sc = ax.scatter(1-df_results['Qb/Q'], 
            mean_r_nd, # df_results['mean hillslope len curvature'][ids], #
            # color=cols[i], 
            # s=df_params['sigma'][ids],
            c=plot_runs,
            # c=df_results['sat_variable'],
            # vmin=0.0,
            # vmax=1.0,
            cmap='plasma',
            )
ax.scatter(1-df.loc['Qb/Q'], mean_r_nd4)        
# ax.legend(frameon=False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((0.03,1))
ax.set_ylim((2,200))
ax.set_xlabel(r'$ \langle Q_f \rangle / \langle Q \rangle$')
ax.set_ylabel(r'$\overline{Z} / h_g$')

fig.tight_layout()
# plt.savefig('%s/%s/relief_qfi_%s_sat.pdf'%(directory, base_output_path, out_names[case]), dpi=300) #_sat


#%% slope histogram

col = cm.get_cmap('viridis', 4)
cols = col(np.linspace(0.2,1.0,4))

fig, ax = plt.subplots(figsize=(6,4.2))
for i, IT in enumerate(plot_runs):
  
    grid = from_netcdf('%s/%s/grid_%d_%d.nc'%(directory, base_output_path, ID, IT))
    slope = grid.at_node['slope_D4'][grid.core_nodes]
    
    plt.hist(slope, bins=50, alpha= 0.5, density=True, color=cols[i], histtype='stepfilled', edgecolor='k')
plt.xlabel(r'$|\nabla z|$')
plt.ylabel('pdf')
#%% different timesteps: Asat-Qb

ID = 4

paths = glob.glob('%s/%s/grid_%d_*'%(directory, base_output_path, ID))
iterations = sorted([int(x.split('_')[-1][:-3]) for x in paths])

ameans = np.zeros(4)
qbmeans = np.zeros(4)
qmeans = np.zeros(4)

cmap1 = copy.copy(cm.viridis)
cmap1.set_bad(cmap1(0))
letters = ['A', 'B', 'C', 'D']
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8,2))
for i, IT in enumerate(iterations):

    
    # df = pickle.load(open('%s/%s/q_s_dt_ID_%d.p'%(directory, base_output_path, i), 'rb'))
    df = pd.read_csv('%s/%s/q_s_dt_ID_%d_%d.csv'%(directory, base_output_path, ID, IT))
    grid = from_netcdf('%s/%s/grid_%d_%d.nc'%(directory, base_output_path, ID, IT))
   
    Atot = np.sum(grid.cell_area_at_node[grid.core_nodes])
    Qb = df['qb']/(Atot*df_params['p'][i])
    
    Q = df['qs_star']
    S = df['S_star']
    qs_cells = df['qs_cells']/grid.number_of_cells #same as number of core nodes
    r = df['r']
    ibar = (df_params['ds'][i]/df_params['tr'][i])*np.sum(grid.cell_area_at_node)
    rstar = r/ibar

    sort = np.argsort(S)

    Amean = np.sum(qs_cells*df['dt'])/np.sum(df['dt'])
    Qbmean = np.sum(Qb*df['dt'])/np.sum(df['dt'])
    ameans[i] = Amean
    qbmeans[i] = Qbmean
    qmeans[i] = np.sum(Q*df['dt'])/np.sum(df['dt'])

    ind = 1000
    
    if i == 0:
        axs[i].axhline(Amean, color='b', linestyle='--', linewidth=1, label=r"$\langle A_{sat}^* \rangle$")
        axs[i].axvline(Qbmean, color='r', linestyle='--', linewidth=1, label=r"$\langle Q_b^* \rangle$")
    else:
        axs[i].axhline(Amean, color='b', linestyle='--', linewidth=1)
        axs[i].axvline(Qbmean, color='r', linestyle='--', linewidth=1)

    sc = axs[i].scatter(Q[ind:], 
                            qs_cells[ind:], 
                            color='lightgray', 
                            s=4, 
                            alpha=0.05,
                            rasterized=True)
    
    sc = axs[i].scatter(Qb[ind:], #[sort[ind:]] 
                            qs_cells[ind:], 
                            c=S[ind:], 
                            s=4, 
                            alpha=0.2, 
                            # vmin=0.0,
                            # vmax=1.0,
                            norm=colors.LogNorm(vmin=1e-3, vmax=1), 
                            cmap=cmap1,
                            rasterized=True)
    
    axs[i].text(0.05, 
                    0.95, 
                    letters[i], 
                    transform=axs[i].transAxes, 
                    fontsize=10, 
                    verticalalignment='top',
                    )
    axs[i].ticklabel_format(style='sci')
    axs[i].tick_params(axis='both', which='major')
    axs[i].set_ylim((1e-3,1))
    axs[i].set_xlim((1e-3,200))
    axs[i].set_yscale('log')
    axs[i].set_xscale('log')
    # if m//nrows != nrows-1:
    #     axs[m, n].set_xticklabels([])
    # if n%ncols != 0:
    #     axs[m, n].set_yticklabels([])
    
fig.subplots_adjust(right=0.75, bottom=0.3, wspace=0.45)
rect_cb = [0.8, 0.3, 0.03, 0.6]
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, orientation="vertical")

cbar.set_label(label='$S^*$', size=14)
cbar.set_alpha(1.0)
cbar.draw_all()

axs[0].set_ylabel('$A^*_{sat}$', size=14)
axs[0].set_xlabel('$Q^*$', size=14)
axs[0].legend(loc=8, facecolor='w', edgecolor='w')

plt.savefig('%s/%s/Q_sat_S_%s_%d.pdf'%(directory, base_output_path, base_output_path, ID ), transparent=True, dpi=300)


qbchange = (qbmeans[-1]-qbmeans[0])/qbmeans[0]
achange = (ameans[-1]-ameans[0])/ameans[0]
qchange = (qmeans[-1]-qmeans[0])/qmeans[0]


#%% different timesteps: sat class

always_sat = np.zeros(4)
variable_sat = np.zeros(4)
letters = ['A', 'B', 'C', 'D']
labels = ["dry", "variable", "wet"]    
L1 = ["peru", "dodgerblue", "navy"]
L2 = ['r', 'g', 'b']
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8,2))
for i, IT in enumerate(iterations):
    
    grid = from_netcdf('%s/%s/grid_%d_%d.nc'%(directory, base_output_path, ID, IT))
    sat_class = grid.at_node['saturation_class']
    
    always_sat[i] = np.sum(sat_class==1)/np.number_of_core_nodes
    
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    cmap = colors.ListedColormap(L1)
    norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)
    fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    im = axs[i].imshow(sat_class.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap=cmap,
                         norm=norm,
                         )
    axs[i].text(0.05, 
                0.95, 
                letters[i], 
                transform=axs[i].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                )  

    axs[i].set_xticklabels([])
    axs[i].set_yticklabels([])

fig.subplots_adjust(right=0.75, bottom=0.3, wspace=0.35)
rect_cb = [0.8, 0.4, 0.03, 0.4]
cbar_ax = plt.axes(rect_cb)
cbar = fig.colorbar(im, cax=cbar_ax, format=fmt, ticks=np.arange(0,3))

plt.savefig('%s/%s/sat_zones_%s_%d.pdf'%(directory, base_output_path, base_output_path, ID), transparent=True, dpi=300)

#%% different timesteps: hillshades

letters = ['A', 'B', 'C', 'D']
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(8,2))
for i, IT in enumerate(iterations):

    grid = from_netcdf('%s/%s/grid_%d_%d.nc'%(directory, base_output_path, ID, IT))
    # grid = read_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 
    ls = LightSource(azdeg=135, altdeg=45)
    axs[i].imshow(
                    ls.hillshade(elev.reshape(grid.shape).T, 
                        vert_exag=1, 
                        dx=grid.dx, 
                        dy=grid.dy), 
                    origin="lower", 
                    extent=(x[0], x[-1], y[0], y[-1]), 
                    cmap='gray',
                    )
    axs[i].tick_params(axis='both', which='major', labelsize=10)
    axs[i].text(0.05, 
                0.95, 
                letters[i], 
                transform=axs[i].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                )   


axs[0].set_ylabel(r'$y/\ell_g$')
# axs[0].set_xlabel(r'$x/\ell_g$')
fig.subplots_adjust(right=0.75, bottom=0.3, wspace=0.35)
plt.savefig('%s/%s/hillshade_%s_%d.pdf'%(directory, base_output_path, base_output_path, ID), transparent=True, dpi=300)


#%%