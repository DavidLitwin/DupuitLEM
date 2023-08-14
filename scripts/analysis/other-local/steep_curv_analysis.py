
#%%
import glob
import numpy as np
import pandas as pd
import copy

from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from landlab.io.netcdf import from_netcdf

directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'stoch_gam_sigma_14' #'stoch_ai_sigma_10' #
model_runs = np.arange(25)
nrows = 5
ncols = 5

save_directory = 'C:/Users/dgbli/Documents/Papers/Ch3_oregon_ridge_soldiers_delight/figures'

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
df_params['lambda'] = (125*df_params['v0']/df_params['lg'])


# for most runs, plot all 
set_name = 'all'
plot_runs = model_runs

# plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)
plot_array = np.flipud(plot_runs.reshape((nrows, ncols))) # note flipped!

#%% DupuitLEM: aggregate view steepness curvature

# nrows=3
# ncols=2
# plot_runs = np.array([0,2,5,24,26,29])
# plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
 
    S = grid.at_node['slope_D8'][grid.core_nodes]
    A = grid.at_node['drainage_area'][grid.core_nodes]
    curvature = grid.at_node['curvature'][grid.core_nodes]
    q = grid.at_node['surface_water_effective__discharge'][grid.core_nodes] 
    qstar = q/(df_params['p'][i]*A)
    c = np.argsort(qstar)
    # make dimensionless versions
    a_star = (A/grid.dx)/lg
    S_star = S*lg/hg
    steepness_star = np.sqrt(a_star)*S_star
    curvature_star = curvature*lg**2/hg
    
    x = np.linspace(0,max(steepness_star), 10)
    y_theory = x - 1
    sc = axs[m,n].scatter(steepness_star[c], 
                  curvature_star[c], 
                  s=8, 
                  alpha=0.1, 
                  c=qstar[c], 
                  cmap='plasma',
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
    axs[m,n].set_ylim((1.05*np.min(curvature_star), 1.05*np.max(curvature_star)))
    # axs[m,n].set_title(r"$\gamma=%.2f$, $\lambda=%.2f$"%(df_params['gam'][i], df_params['lam'][i]), fontsize=9)

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

# nrows=3
# ncols=2
# plot_runs = np.array([0,2,5,24,26,29])
# plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
    curvature = grid.at_node['curvature'][grid.core_nodes]
    S4 = grid.at_node['slope_D4'][grid.core_nodes]
    twi = grid.at_node['topographic__index_D4'][grid.core_nodes]/np.cos(np.arctan(S4))**2
    qstar = grid.at_node['surface_water_effective__discharge']/(df_params['p'][i]*grid.at_node['drainage_area'])
    c = np.argsort(qstar)

    # make dimensionless versions
    twi_star = twi*hg/lg**2
    curvature_star = curvature*lg**2/hg
    
    sc = axs[m,n].scatter(twi_star[c], 
                  curvature_star[c], 
                  s=8, 
                  alpha=0.1, 
                  c=qstar[c], 
                  cmap='plasma',
                  vmin=0.0,
                  vmax=0.2,
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
    # axs[m,n].set_title(r"$\gamma=%.2f$, $\lambda=%.2f$"%(df_params['gam'][i], df_params['lam'][i]), fontsize=9)

axs[-1,0].set_xlabel(r"$a'/(|\nabla' z'| \cos^2 \theta)$")
axs[-1,0].set_ylabel(r"$\nabla'^{2} z'$")
fig.subplots_adjust(right=0.75, hspace=0.3)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
cbar = fig.colorbar(sc, cax=ax_cb, label=r'$Q^*$', orientation="vertical")
cbar.solids.set_edgecolor("face")
# plt.tight_layout()
plt.savefig('%s/curv_twi_qstar_%s.png'%(save_directory, base_output_path), dpi=300)

#%% DupuitLEM: aggregate view steepness curvature

# nrows=3
# ncols=2
# plot_runs = np.array([0,2,5,24,26,29])
# plot_array = np.flipud(plot_runs.reshape((ncols, nrows)).T)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
 
    S = grid.at_node['slope_D8'][grid.core_nodes]
    A = grid.at_node['drainage_area'][grid.core_nodes]
    curvature = grid.at_node['curvature'][grid.core_nodes]
    q = grid.at_node['surface_water_effective__discharge'][grid.core_nodes] 
    qstar = q/(df_params['p'][i]*A)
    # make dimensionless versions
    a_star = (A/grid.dx)/lg
    S_star = S*lg/hg
    steepness_star = np.sqrt(a_star)*S_star
    curvature_star = curvature*lg**2/hg
    
    labels = ["dry", "variable", "wet"] 
    L1 = ["peru", "dodgerblue", "navy"]
    cmap = colors.ListedColormap(L1)
    norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)
    fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    sc = axs[m,n].scatter( qstar, #q/(np.sqrt(A)*df_params['p'][i]), #q,
                  S_star, 
                  s=8, 
                  alpha=0.1, 
                  c=curvature_star, 
                  norm=norm, #colors.TwoSlopeNorm(0,vmin=-5,vmax=5),
                  cmap=cmap, #'bwr'
                  )
    axs[m,n].text(0.1, 
                0.2, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                ) 
    # axs[m,n].plot(np.geomspace(1e-5,1), 0.025*np.geomspace(1e-5,1)**(-0.5), label=r'$Q^{-1/2}$' ) #Q
    axs[m,n].plot(np.geomspace(1e-1,1e4), 10*np.geomspace(1e-1,1e4)**(-1) , 'k--', label='1:1') #Q
    axs[m,n].set_yscale('log')
    axs[m,n].set_xscale('log')
    # axs[m,n].set_xlim((1e-5, 1)) #Q
    # axs[m,n].set_xlim((1e-2, 1)) #Q*
    axs[m,n].set_xlim((1e-1, 1e4)) #Q/(sqrt(a)p)

    axs[m,n].set_ylim((1e-2, 20))

    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])

# axs[-1,0].set_xlabel(r'$Q$')
# axs[-1,0].set_xlabel(r'$Q^*$')
axs[-1,0].set_xlabel(r'$Q/(p\sqrt{A})$')
axs[-1,0].set_ylabel(r"$|\nabla' z'|$")
fig.subplots_adjust(right=0.75, hspace=0.3)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
# cbar = fig.colorbar(sc, cax=ax_cb, label=r"$\nabla'^{2} z'$", orientation="vertical")
cbar = fig.colorbar(sc, cax=ax_cb, format=fmt, ticks=np.arange(0,3))
cbar.solids.set_edgecolor("face")
axs[0,0].legend(frameon=False)
# plt.savefig('%s/S_qstar_curv_%s.png'%(save_directory, base_output_path), dpi=300)
# plt.savefig('%s/S_q_curv_%s.png'%(save_directory, base_output_path), dpi=300)


# %% Q* with increasing area

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11,8))
for i in plot_runs:
    m = np.where(plot_array==i)[0][0]
    n = np.where(plot_array==i)[1][0]
    
    grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    lg = df_params['lg'][i]
    hg = df_params['hg'][i]
 
    S = grid.at_node['slope_D8'][grid.core_nodes]
    A = grid.at_node['drainage_area'][grid.core_nodes]
    curvature = grid.at_node['curvature'][grid.core_nodes]
    q = grid.at_node['surface_water_effective__discharge'][grid.core_nodes] 
    # qstar = q/(df_params['p'][i]*A)
    
    qstar = grid.at_node['surface_water_effective__discharge']/(df_params['p'][i]*grid.at_node['drainage_area'])


    # make dimensionless versions
    a_star = (A/grid.dx)/lg
    S_star = S*lg/hg
    steepness_star = np.sqrt(a_star)*S_star
    curvature_star = curvature*lg**2/hg
    
    labels = ["dry", "variable", "wet"] 
    L1 = ["peru", "dodgerblue", "navy"]
    cmap = colors.ListedColormap(L1)
    norm = colors.BoundaryNorm(np.arange(-0.5, 3), cmap.N)
    fmt = ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    sc = axs[m,n].scatter(a_star, # 
                  qstar[grid.core_nodes], 
                  s=8, 
                  alpha=0.1, 
                  c=curvature_star, 
                  norm=norm, #colors.TwoSlopeNorm(0,vmin=-5,vmax=5),
                  cmap=cmap, #'bwr'
                  )
    axs[m,n].text(0.8, 
                0.2, 
                str(i), 
                transform=axs[m,n].transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(ec='w',fc='w')
                ) 
    axs[m,n].axhline(y=df_params['ai'][i], linestyle='--', color='k', label='Ai')
    axs[m,n].axhline(y=1-df_results['(P-Q-Qgw)/P'][i], linestyle='--', color='r', label='1-AET/P')
    # axs[m,n].set_yscale('log')
    axs[m,n].set_xscale('log')
    axs[m,n].set_ylim((1e-2, 1)) #Q*
    axs[m,n].set_xlim((1, 2e4))

    if m != nrows-1:
        axs[m, n].set_xticklabels([])
    if n != 0:
        axs[m, n].set_yticklabels([])

axs[-1,0].set_ylabel(r'$Q^*$')
axs[-1,0].set_xlabel(r"$a'$")
fig.subplots_adjust(right=0.75, hspace=0.3)
rect_cb = [0.8, 0.35, 0.03, 0.3]
ax_cb = plt.axes(rect_cb)
# cbar = fig.colorbar(sc, cax=ax_cb, label=r"$\nabla'^{2} z'$", orientation="vertical")
cbar = fig.colorbar(sc, cax=ax_cb, format=fmt, ticks=np.arange(0,3))
cbar.solids.set_edgecolor("face")
axs[0,0].legend(frameon=False)
# plt.savefig('%s/astar_qstar_%s.png'%(save_directory, base_output_path), dpi=300)

# %%
#%% individual runoff

i = 10

fig, axs = plt.subplots(figsize=(4,4)) 
grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
qstar = grid.at_node['surface_water_effective__discharge']/(df_params['p'][i]*grid.at_node['drainage_area'])

A = grid.at_node['drainage_area']


lg = df_params['lg'][i]
hg = df_params['hg'][i]
dx = grid.dx/df_params['lg'][i]
y = np.arange(grid.shape[0] + 1) * dx - dx * 0.5
x = np.arange(grid.shape[1] + 1) * dx - dx * 0.5

im = axs.imshow(qstar.reshape(grid.shape).T, 
                     origin="lower", 
                     extent=(x[0], x[-1], y[0], y[-1]), 
                      cmap='plasma',
                     # norm=colors.LogNorm(vmin=1e-3, vmax=1),
                     vmin=0.0,
                     vmax=1.0,
                     interpolation = 'none'
                     )
axs.axis('off')
fig.colorbar(im, label=r'$Q^*$')
# plt.savefig('%s/%s/qstar_%s_%d.png'%(directory, base_output_path, base_output_path, i), dpi=300, transparent=True)

# %%
