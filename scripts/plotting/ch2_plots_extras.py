# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:39:07 2022

@author: dgbli
"""
import glob
import numpy as np
import pandas as pd
import copy
import linecache

from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LightSource
from landlab.io.netcdf import from_netcdf
plt.rc('text', usetex=True)


directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'stoch_gam_sigma_14' #'stoch_ai_sigma_10' #
model_runs = np.arange(25)

#%% load results and parameters

dfs = []
for ID in model_runs:
    try:
        df = pd.read_csv('%s/%s/output_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    except FileNotFoundError:
        df =  pd.DataFrame(columns=df.columns)
    dfs.append(df)
df_results = pd.concat(dfs, axis=1, ignore_index=True).T

dfs = []
for ID in model_runs:
    df = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)
    dfs.append(df)
df_params = pd.concat(dfs, axis=1, ignore_index=True).T
df_params.to_csv('%s/%s/params.csv'%(directory,base_output_path), index=True, float_format='%.3e')

# for most runs, plot all 
set_name = 'all'
plot_runs = model_runs
nrows = 5
ncols = 5
# plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)
plot_array = np.flipud(plot_runs.reshape((nrows, ncols))) # note flipped!


#%% budyko baseflow ai-sigma

alt = (df_results['cum_precip']-df_results['cum_recharge'])/df_results['cum_precip']
ai = (df_results['cum_pet']/df_results['cum_precip'])

sigma = np.unique(df_params['sigma'])
viridis = cm.get_cmap('viridis', len(sigma))
vir = viridis.colors

size = 20
fig, axs = plt.subplots(ncols=2, figsize=(9,3))
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[0].plot(ai[cond], alt[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    axs[0].scatter(ai[cond], alt[cond], color=vir[j], alpha=1.0, s=size, zorder=100)
axs[0].plot([0,1], [0,1], 'k--')
axs[0].plot([1,2], [1,1], 'k--')
axs[0].set_xlabel('PET/P')
axs[0].set_ylabel(r'$(P-R)/P$')

qb = df_results['qb_tot']/df_results['cum_runoff']
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[1].plot(alt[cond], qb[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    axs[1].scatter(alt[cond], qb[cond], color=vir[j], alpha=1.0, s=size, zorder=101, label=r'$\sigma$ = %.1f'%sig)
axs[1].set_xlabel(r'$(P-R)/P$')
axs[1].set_ylabel(r'$Q_b/Q$')

axs[1].legend(frameon=False)
fig.tight_layout()
# plt.savefig('%s/%s/budyko_et_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)

#%% budyko - baseflow gam-sigma

alt = (df_results['cum_precip']-df_results['cum_recharge'])/df_results['cum_precip']
ai = (df_results['cum_pet']/df_results['cum_precip'])
gam = df_params['gam']

sigma = np.unique(df_params['sigma'])
viridis = cm.get_cmap('viridis', len(sigma))
vir = viridis.colors

size = 20
fig, axs = plt.subplots(ncols=2, figsize=(9,3))
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[0].plot(gam[cond], alt[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    axs[0].scatter(gam[cond], alt[cond], color=vir[j], alpha=1.0, s=size, zorder=100)
axs[0].axhline(y=max(ai), label='PET/P', linestyle='--', color='k')
# axs[0].plot([0,1], [0,1], 'k--')
# axs[0].plot([1,2], [1,1], 'k--')
axs[0].set_xlabel(r'$\gamma$')
axs[0].set_ylabel(r'$(P-R)/P$')
axs[0].legend(frameon=False)

qb = df_results['qb_tot']/df_results['cum_runoff']
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[1].plot(alt[cond], qb[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    axs[1].scatter(alt[cond], qb[cond], color=vir[j], alpha=1.0, s=size, zorder=101)
axs[1].set_xlabel(r'$(P-R)/P$')
axs[1].set_ylabel(r'$Q_b/Q$')

axs[1].legend(frameon=False)
fig.tight_layout()
# plt.savefig('%s/%s/budyko_et_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)

#%% budyko - paper draft 1

df_results['cum_et_calc'] = df_results['cum_precip'] - df_results['cum_runoff'] - df_results['cum_gw_export']

alt = (df_results['cum_precip']-df_results['cum_recharge'])/df_results['cum_precip']
ai = (df_results['cum_pet']/df_results['cum_precip'])

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
axs[0].set_xlabel(r'$\langle PET \rangle / \langle P \rangle$')
axs[0].set_ylabel(r'$(\langle ET \rangle)/\langle P \rangle$')

qb = df_results['qb_tot']/df_results['cum_precip']
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[1].plot(alt[cond], qb[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    axs[1].scatter(alt[cond], qb[cond], color=vir[j], alpha=1.0, s=size, zorder=101)
axs[1].set_xlabel(r'$(\langle P \rangle-\langle R \rangle)/\langle P \rangle$')
axs[1].set_ylabel(r'$\langle Q_b \rangle /\langle P \rangle$')

qe = df_results['qe_tot']/df_results['cum_precip']
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[2].plot(alt[cond], qe[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    sc = axs[2].scatter(alt[cond], qe[cond], color=vir[j], alpha=1.0, s=size, zorder=102, label=r'$\sigma$ = %.1f'%sig)
axs[2].set_xlabel(r'$(\langle P \rangle-\langle R \rangle)/\langle P \rangle$')
axs[2].set_ylabel(r'$\langle Q_f \rangle /\langle P \rangle$')

axs[2].legend(frameon=False)
fig.tight_layout()
# plt.savefig('%s/%s/budyko_et_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)

#%% Budyko - gam sigma - paper draft 1

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
axs[0].set_ylabel(r'$(\langle P \rangle-\langle R \rangle)/\langle P \rangle$')
axs[0].legend(frameon=False)

qb = df_results['qb_tot']/df_results['cum_precip']
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[1].plot(alt[cond], qb[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    axs[1].scatter(alt[cond], qb[cond], color=vir[j], alpha=1.0, s=size, zorder=101)
axs[1].set_xlabel(r'$(\langle P \rangle-\langle R \rangle)/\langle P \rangle$')
axs[1].set_ylabel(r'$\langle Q_b \rangle /\langle P \rangle$')

qe = df_results['qe_tot']/df_results['cum_precip']
for j, sig in enumerate(sigma):
    cond = df_params['sigma'] == sig
    axs[2].plot(alt[cond], qe[cond], alpha=0.5, color=vir[j], linewidth=1.0)
    sc = axs[2].scatter(alt[cond], qe[cond], color=vir[j], alpha=1.0, s=size, zorder=102, label=r'$\sigma$ = %.1f'%sig)
axs[2].set_xlabel(r'$(\langle P \rangle-\langle R \rangle)/\langle P \rangle$')
axs[2].set_ylabel(r'$\langle Q_f \rangle /\langle P \rangle$')

axs[2].legend(frameon=False)
fig.tight_layout()
plt.savefig('%s/%s/budyko_et_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)

#%% cross sections

# plot_runs:
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    hg = df_params['hg'][i] # all the same hg and lg
    lg = df_params['lg'][i]
    b = df_params['b'][i]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    base = grid.at_node['aquifer_base__elevation']
    wt_high = grid.at_node['wtrel_mean_end_storm']*b + base
    wt_low = grid.at_node['wtrel_mean_end_storm']*b + base
    middle_row = np.where(grid.x_of_node == np.median(grid.x_of_node))[0][1:-1]
    
    y = grid.y_of_node[middle_row]/lg
    axs[m,n].fill_between(y,elev[middle_row]/hg,base[middle_row]/hg,facecolor=(198/256,155/256,126/256) )
    axs[m,n].fill_between(y,wt_high[middle_row]/hg,base[middle_row]/hg,facecolor=(145/256,176/256,227/256), alpha=0.5)
    axs[m,n].fill_between(y,wt_low[middle_row]/hg,base[middle_row]/hg,facecolor=(145/256,176/256,227/256), alpha=0.5)
    axs[m,n].fill_between(y,base[middle_row]/hg,np.zeros_like(base[middle_row]),facecolor=(111/256,111/256,111/256))
    #ax.set_ylim((0,30))
    # ax.set_xlim((0,995))
    
    axs[m,n].set_xticks(range(0,100,25))
    if m != nrows-1:
        axs[m, n].set_xticklabels([])

# axs[-1, 0].set_ylabel(r'$z/h_g$')
# axs[-1, 0].set_xlabel(r'$x/\ell_g$')
# plt.tight_layout()

#%% plot_runs Qstar

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,5)) #(8,5)
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    qstar = grid.at_node['surface_water_effective__discharge']/(df_params['p'][i]*grid.at_node['drainage_area'])
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
                         interpolation = 'none'
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
# axs[-1, 0].set_ylabel(r'$y/\ell_g$')
# axs[-1, 0].set_xlabel(r'$x/\ell_g$')
plt.savefig('%s/%s/qstar_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)


#%% recharge vs topographic index

cmap1 = copy.copy(cm.cividis)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8,5)) #(10,7)
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    hg = df_params['hg'][i]
    lg = df_params['lg'][i]
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    S4 = grid.at_node['slope_D4']
    twi = grid.at_node['topographic__index_D4']/np.cos(np.arctan(S4))**2
    twi_star = twi*hg/lg**2
    recharge = grid.at_node['recharge_rate_mean_storm']/(df_params['ds'][i]/df_params['tr'][i])    
    wt = grid.at_node['wtrel_mean_end_interstorm']
    
    sc = axs[m, n].scatter(np.log(twi_star[grid.core_nodes]), 
                           recharge[grid.core_nodes], 
                           c=wt[grid.core_nodes], 
                           cmap=cmap1,
                           vmin=0.0, 
                           vmax=1.0, 
                           s=5, 
                           alpha=0.2,
                           rasterized=True)
    axs[m, n].text(0.05, 
                    0.95, 
                    str(i), 
                    transform=axs[m, n].transAxes, 
                    fontsize=8, 
                    verticalalignment='top',
                    )
    # axs[m, n].ticklabel_format(style='sci')
    # axs[m, n].tick_params(axis='both', which='major', labelsize=6)
    axs[m ,n].set_ylim((0,1))
    axs[m ,n].set_xlim((-3,17))

    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    
fig.subplots_adjust(right=0.75, hspace=0.3, wspace=0.3)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, orientation="vertical")
cbar.set_label(label=r"$\overline{z_{wt}}/b$", size=14)
cbar.solids.set_edgecolor("face")

axs[-1, 0].set_ylabel(r"$\bar{r}/\bar{i}$")
axs[-1, 0].set_xlabel(r"$\log(TI^*)$")

plt.savefig('%s/%s/R_twi_nd_%s.pdf'%(directory, base_output_path, set_name), dpi=300)

#%% sat zones single

i = 7

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
L2 = ['r', 'g', 'b']
cmap = colors.ListedColormap(L1)
norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)
fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

fig, axs = plt.subplots(figsize=(4,4))
im = axs.imshow(sat_class.reshape(grid.shape).T, 
                     origin="lower", 
                     extent=(x[0], x[-1], y[0], y[-1]), 
                     cmap=cmap,
                     norm=norm,
                     interpolation=None,
                     )
axs.text(0.03, 
            0.97, 
            str(i), 
            transform=axs.transAxes, 
            fontsize=12, 
            verticalalignment='top',
            bbox=dict(ec='w',fc='w')
            )  
axs.set_xticklabels([])
axs.set_yticklabels([])
plt.savefig('%s/%s/sat_zones_%s_%d.png'%(directory, base_output_path, base_output_path, i), dpi=300)

#%% select saturtion discharge

# plot_runs_0 = np.array([0, 2, 14, 17, 28, 31])
plot_runs_0 = np.array([0, 4, 10, 14, 20, 24])
plot_array_0 = np.flipud(plot_runs_0.reshape((3,2)))

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(5,6))
for i in plot_runs_0:
    m = np.where(plot_array_0==i)[0][0]
    n = np.where(plot_array_0==i)[1][0]
    
    # df = pickle.load(open('%s/%s/q_s_dt_ID_%d.p'%(directory, base_output_path, i), 'rb'))
    df = pd.read_csv('%s/%s/q_s_dt_ID_%d.csv'%(directory, base_output_path, i))
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    ps = np.sum(grid.at_node['saturation_class'] == 2)/grid.number_of_core_nodes
    
    Q = df['qs_star']
    S = df['S_star']
    qs_cells = df['sat_nodes']/grid.number_of_cells #same as number of core nodes
    sc = axs[m, n].scatter(Q[1000:], 
                           qs_cells[1000:], 
                           c=S[1000:], 
                           s=5, 
                           alpha=0.2, 
                           vmin=0.0, 
                           vmax=1.0, 
                           cmap='cividis')
    axs[m,n].axhline(ps)
    axs[m, n].text(0.05, 
                    0.95, 
                    str(i), 
                    transform=axs[m, n].transAxes, 
                    fontsize=10, 
                    verticalalignment='top',
                    )
    axs[m, n].ticklabel_format(style='sci')
    # axs[m, n].tick_params(axis='both', which='major', labelsize=6)
    axs[m ,n].set_ylim((1e-3,1))
    axs[m ,n].set_xlim((1e-3,200))
    axs[m ,n].set_yscale('log')
    axs[m ,n].set_xscale('log')
    # if m//nrows != nrows-1:
    #     axs[m, n].set_xticklabels([])
    # if n%ncols != 0:
    #     axs[m, n].set_yticklabels([])
axs[-1, 0].set_ylabel('$A^*_{sat}$', size=14)
axs[-1, 0].set_xlabel('$Q^*$', size=14)
fig.tight_layout()

fig.subplots_adjust(right=0.75, hspace=0.3, wspace=0.4)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, orientation="vertical")
cbar.set_label(label='$S^*$', size=14)
cbar.solids.set_edgecolor("face")


# plt.savefig('%s/%s/Q_sat_S_%s_subset.png'%(directory, base_output_path, set_name), dpi=300)

#%% discussion sat areas

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

ID = 4

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
# vals = [0.02, 0.05, 0.1, 0.5]
vals = [0.007, 0.05, 0.1, 0.5]
vals_closest = []
idxs = []
lines = np.zeros((len(vals), len(r)))
for i, val in enumerate(vals):
    print(i, val)
    valc, idx = find_nearest(sat_mean, val)
    idxs.append(idx)
    vals_closest.append(valc)
    
    line = linecache.getline('%s/%s/sat_%d.csv'%(directory, base_output_path, ID), idx)
    lines[i,:] = np.array(line.split(','),int)


# all together in gridspec 
elev = grid.at_node['topographic__elevation']
dx = grid.dx/df_params['lg'][ID]
y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5

ls = LightSource(azdeg=135, altdeg=45)
cmap = colors.ListedColormap(['blue'])
fig = plt.figure(constrained_layout=True, figsize=(8,6))
ind = 0

subax = []
gs = GridSpec(6, 8, figure=fig)

for j in range(len(vals)):
    ax = fig.add_subplot(gs[4:, j*2:j*2+2])
    subax.append(ax)
    
    
    ax.imshow(
        ls.hillshade(elev.reshape(grid.shape).T, 
            vert_exag=1, 
            dx=grid.dx, 
            dy=grid.dy), 
        origin="lower", 
        extent=(x[0], x[-1], y[0], y[-1]), 
        cmap='gray',
        )
    sat = np.ma.masked_where(lines[j,:]==0, lines[j,:])
    ax.imshow(sat.reshape(grid.shape).T, 
            origin="lower", 
            extent=(x[0], x[-1], y[0], y[-1]),
            alpha=0.5,
            cmap=cmap,
            interpolation="none"
            )
    ax.set_xticks([0,100,200])
    ax.set_yticks([0,100,200])
    # ax.set_xlim((x[1],x[-2]))
    # ax.set_ylim((y[1],y[-2]))
    ax.set_title(r'$A^*_{sat}=%.2f$'%vals_closest[j])
    
    if j == 0:
        ax.set_xlabel(r'$x/\ell_g$')
        ax.set_ylabel(r'$y/\ell_g$')
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
ax0 = fig.add_subplot(gs[0:4, 2:6])
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
ax0.set_ylim((3e-3,1))
ax0.set_xlim((1e-4,10))
ax0.set_yscale('log')
ax0.set_xscale('log')
ax0.set_ylabel('$A^*_{sat}$', size=16)
ax0.set_xlabel('$Q_b^*$', size=16)

    

plt.show()
plt.savefig('%s/%s/sat_areas_detail.pdf'%(directory, base_output_path), dpi=300)

#%% cross analysis
base_output_path = 'stoch_gam_sigma_8' #'stoch_ai_sigma_4' 
grid_id_l = 0
grid_id_h = 4
model_runs = np.arange(0,5)

cross_folder_l = 'cross_%d'%grid_id_l
cross_folder_h = 'cross_%d'%grid_id_h

dfs = []
for ID in model_runs:
    try:
        df = pd.read_csv('%s/%s/%s/output_%d_x_%d.csv'%(directory,base_output_path, cross_folder_l, grid_id_l, ID), index_col=0)
    except FileNotFoundError:
        df =  pd.DataFrame(columns=df.columns)
    dfs.append(df)
df_results_l = pd.concat(dfs, axis=1, ignore_index=False).T

dfs = []
for ID in model_runs:
    df = pd.read_csv('%s/%s/%s/params_%d_x_%d.csv'%(directory,base_output_path, cross_folder_l, grid_id_l, ID), index_col=0)
    dfs.append(df)
df_params_l = pd.concat(dfs, axis=1, ignore_index=False).T
df_params_l.to_csv('%s/%s/%s/params.csv'%(directory,base_output_path, cross_folder_l), index=True, float_format='%.3e')

#######

dfs = []
for ID in model_runs:
    try:
        df = pd.read_csv('%s/%s/%s/output_%d_x_%d.csv'%(directory,base_output_path, cross_folder_h, grid_id_h, ID), index_col=0)
    except FileNotFoundError:
        df =  pd.DataFrame(columns=df.columns)
    dfs.append(df)
df_results_h = pd.concat(dfs, axis=1, ignore_index=False).T

dfs = []
for ID in model_runs:
    df = pd.read_csv('%s/%s/%s/params_%d_x_%d.csv'%(directory,base_output_path, cross_folder_h, grid_id_h, ID), index_col=0)
    dfs.append(df)
df_params_h = pd.concat(dfs, axis=1, ignore_index=False).T
df_params_h.to_csv('%s/%s/%s/params.csv'%(directory,base_output_path, cross_folder_h), index=True, float_format='%.3e')


#%% cross analysis Budyko

# df_results_h['(P-R)/P'] = (df_results_h.cum_precip - df_results_h.cum_recharge)/df_results_h.cum_precip
# df_results_l['(P-R)/P'] = (df_results_l.cum_precip - df_results_l.cum_recharge)/df_results_l.cum_precip
# df_results['(P-R)/P'] = (df_results.cum_precip - df_results.cum_recharge)/df_results.cum_precip

# df_results_h['(P-R+E)/P'] = (df_results_h.cum_precip - df_results_h.cum_recharge - df_results_h.cum_extraction )/df_results_h.cum_precip
# df_results_l['(P-R+E)/P'] = (df_results_l.cum_precip - df_results_l.cum_recharge - df_results_l.cum_extraction )/df_results_l.cum_precip
# df_results['(P-R+E)/P'] = (df_results.cum_precip - df_results.cum_recharge - df_results.cum_extraction )/df_results.cum_precip


fig, ax = plt.subplots(figsize=(5,4)) #'(P-Q-Qgw)/P'
sc = ax.scatter(df_params['ai'][model_runs], df_results['(P-Q-Qgw)/P'][model_runs], color='k', alpha=1.0, label='orig')
sc = ax.scatter(df_params_l['ai'], df_results_l['(P-Q-Qgw)/P'], color='b', alpha=1.0, label='low Ai topog')
sc = ax.scatter(df_params_h['ai'], df_results_h['(P-Q-Qgw)/P'], color='r', alpha=1.0, label='high Ai topog')
ax.plot([0,1], [0,1], 'k--')
ax.plot([1,2], [1,1], 'k--')
ax.set_xlabel('Ai')
ax.set_ylabel('(P-Q-Qgw)/P')
fig.tight_layout()
plt.legend()
# plt.savefig('%s/%s/budyko_%s_cross.png'%(directory, base_output_path, base_output_path), dpi=300)

#%% cross analysis Lvovich: Ai sigma

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,6))
sc = ax[0].scatter(df_params.ai[model_runs], df_results['W/P'][model_runs], color='k', label='orig')
sc = ax[0].scatter(df_params_l.ai, df_results_l['W/P'], color='b', label='low Ai topog')
sc = ax[0].scatter(df_params_h.ai, df_results_h['W/P'], color='r', label='high Ai topog')
ax[0].set_ylim((0.2, 1.0))
ax[0].set_xlabel(r'Ai')
ax[0].set_ylabel(r'$W/P$')

# ax[1].scatter(df_params.ai, df_results['Qb/W'], c=df_params.sigma)
ax[1].scatter(df_params.ai[model_runs], df_results['Qb/W'][model_runs], color='k', label='orig')
ax[1].scatter(df_params_l.ai, df_results_l['Qb/W'], color='b', label='low Ai topog')
ax[1].scatter(df_params_h.ai, df_results_h['Qb/W'], color='r', label='high Ai topog')
ax[1].set_ylim((0.0, 0.8))
ax[1].set_xlabel(r'Ai')
ax[1].set_ylabel(r'$Q_b/W$')
ax[1].legend(frameon=False)

# fig.tight_layout()
plt.savefig('%s/%s/lvovich_type_%s_cross.png'%(directory, base_output_path, base_output_path), dpi=300)

#%% 
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,6))
sc = ax[0].scatter(df_params.gam[model_runs], df_results['W/P'][model_runs], color='k', label='orig', s=100)
sc = ax[0].scatter(df_params_l.gam, df_results_l['W/P'], color='b')
sc = ax[0].scatter(df_params_h.gam, df_results_h['W/P'], color='r')
ax[0].set_ylim((0.2, 1.0))
ax[0].set_xlabel(r'Ai')
ax[0].set_ylabel(r'$W/P$')

# ax[1].scatter(df_params.ai, df_results['Qb/W'], c=df_params.sigma)
ax[1].scatter(df_params.gam[model_runs], df_results['Qb/W'][model_runs], color='k', label='orig', s=100)
ax[1].scatter(df_params_l.gam, df_results_l['Qb/W'], color='b', label=r'low $\gamma$ topog')
ax[1].scatter(df_params_h.gam, df_results_h['Qb/W'], color='r', label=r'high $\gamma$ topog')
ax[1].set_ylim((0.0, 0.8))
ax[1].set_xlabel(r'$\gamma$')
ax[1].set_ylabel(r'$Q_b/W$')
ax[1].legend(frameon=False)

# fig.tight_layout()
plt.savefig('%s/%s/lvovich_cross_%d_%d.png'%(directory, base_output_path, grid_id_l, grid_id_h), dpi=300)


#%% plot dimensionless hydrological timeseries and storage-discharge

i=4
# import timeseries and a grid
# df = pickle.load(open('%s/%s/q_s_dt_ID_%d.p'%(directory, base_output_path, i), 'rb'))
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

th = ((df_params['D']*df_params['ne'])/(df_params['ksat']*df_params['U']))
t_star = np.cumsum(df['dt'])/th[ID]

# plt.figure()
# plt.plot(S_star_rec,q_star_rec, '.')
# plt.xlabel('$S^*$ [-]')
# plt.ylabel('$Q^*$ $[-]$')
# plt.xscale('log')
# plt.yscale('log')
# plt.savefig('%s/%s/s_q_nd_%d.png'%(directory, base_output_path,ID))

plt.figure()
plt.plot(t_star,S_star)
plt.xlabel('$t^*$ [-]')
plt.ylabel('$S^*$ [-]')
# plt.savefig('%s/%s/s_nd_%d.png'%(directory, base_output_path,ID))

fig, ax = plt.subplots(figsize=(3.5,2.5))
# plt.figure(figsize=(4,3))
ax.plot(t_star,q_star, color='k', label="$Q^*$")
# plt.plot(t_star, qb_star, color='b', label="$Q_b^*$")
# plt.yscale('log')
# plt.xlabel('$t/t_h$')
# plt.ylabel('$Q^*$')
# plt.xlim((271.6,272.8))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim((1890,1940))
ax.set_ylim((0,12))
# plt.legend(frameon=False)
# plt.tight_layout()
# 
#%% sensitivity of saturation class

# IDs = [0,4,20,24,12]
IDs = [0, 3, 5, 16, 28]
# IDs = [2, 3, 10, 17]

for ID in IDs:
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
    
    plt.figure()
    plt.fill_between(threshold, sat_always, color=L1[2])
    plt.fill_between(threshold, sat_always, sat_variable+sat_always, color=L1[1])
    plt.fill_between(threshold, sat_variable+sat_always, np.ones_like(threshold), color=L1[0])
    plt.title('Saturation Class Sensitivity: ID: %d'%ID)
    # plt.savefig('%s/%s/sat_sensitivity_%d.png'%(directory, base_output_path, ID), dpi=300)
    
    
#%% hand


fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,5)) #(8,5)
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    elev = grid.at_node['topographic__elevation']
    hand = grid.at_node['hand_curvature']
    surf = elev - hand
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    dx = grid.dx/df_params['lg'][i]
    y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
    x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5
 

    im = axs[m, n].imshow(surf.reshape(grid.shape).T, 
                         origin="lower", 
                         extent=(x[0], x[-1], y[0], y[-1]), 
                         cmap='plasma',
                         # vmin=0.0,
                         # vmax=100,
                         interpolation = 'none'
                         )
    # axs[m,n].text(0.05, 
    #             0.95, 
    #             str(i), s
    #             transform=axs[m,n].transAxes, 
    #             fontsize=8, 
    #             verticalalignment='top',
    #             bbox=dict(ec='w',fc='w')
    #             )  
    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])
    
plt.subplots_adjust(left=0.15, right=0.8, wspace=0.15, hspace=0.15)
# plt.tight_layout()
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) #Left, Bottom, Width, Height
fig.colorbar(im, cax=cbar_ax, label=r"Thickness")
# axs[-1, 0].set_ylabel(r'$y/\ell_g$')
# axs[-1, 0].set_xlabel(r'$x/\ell_g$')
# plt.savefig('%s/%s/qstar_%s.pdf'%(directory, base_output_path, base_output_path), dpi=300)


