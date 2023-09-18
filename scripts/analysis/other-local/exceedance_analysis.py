# -*- coding: utf-8 -*-
"""
Run post processing but save the whole grid. Use that grid to derive distributions of runoff.


"""

#%%
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm

from landlab import RasterModelGrid, imshow_grid
from landlab.io.netcdf import to_netcdf, from_netcdf
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    )
from DupuitLEM.auxiliary_models import HydrologyEventVadoseStreamPower, SchenkVadoseModel
from DupuitLEM.grid_functions import bind_avg_exp_ksat, bind_avg_recip_ksat


def get_neighborhood_indices(field, p, n, mask=None):
    """get indices of points in the neighborhood of a certain quantile
    
    field: field to check for quantiles in
    p: percentile
    n: number in neighborhood (odd)
    """

    # mask array if needed
    if mask is not None:
        f = np.sort(field[mask])
    else:
        f = np.sort(field)
    # get single index
    ind_s = np.argwhere(f==np.percentile(f,p, method="nearest"))[0][0]

    # find n closest values
    ns=n//2
    inds_s=np.arange(ind_s-ns,ind_s+ns+1)
    inds_s = inds_s[inds_s>=0] # exclude points out of range (lower)
    inds_s = inds_s[ind_s<len(f)] # exclude points out of range (upper)
    vals = f[inds_s][0]

    # get indices in original array
    inds = [np.argwhere(field==v)[0][0] for v in vals]

    return inds, vals


#%%

# directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
directory = '/Users/dlitwin/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'stoch_gam_sigma_14'
model_runs = np.arange(25)
ID=6

df_params = pd.read_csv('%s/%s/params_ID_%d.csv'%(directory,base_output_path, ID), index_col=0)[str(ID)]

grid = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, ID))
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wtrel = grid.at_node['wtrel_mean_end_interstorm']

# get parameter types right
for ind in df_params.index:
    try:
        df_params[ind] = float(df_params[ind])
    except ValueError:
        df_params[ind] = str(df_params[ind])

ne = df_params['ne'] #drainable porosity [-]
b = df_params['b'] #characteristic depth  [m]
p = df_params['p'] #average precipitation rate [m/s]
tg = df_params['tg']
dtg = df_params['dtg']
hg = df_params['hg']
pet = df_params['pet']
na = df_params['na'] #plant available volumetric water content
tr = df_params['tr'] #mean storm duration [s]
tb = df_params['tb'] #mean interstorm duration [s]
ds = df_params['ds'] #mean storm depth [m]
T_h = 1000*(tr+tb) #20*df_params['Th'] #total hydrological time [s]
sat_cond = 0.025 # distance from surface (units of hg) for saturation


# # change the hydraulic conductivity to depth-dependent
# def ks_func(T, b, d):
#     return T/(d*np.log((b+d)/d))
# df_params['ksat_type'] = 'recip'
# df_params['kdecay'] = 0.2 * df_params['b']
# df_params['ksurface'] = ks_func(df_params['ksat']*df_params['b'], df_params['b'], df_params['kdecay'])


try:
    bc = list(str(df_params['BCs']))
except KeyError:
    bc = None

# hydraulic conductivity
try:
    ksat_type = df_params['ksat_type']

    if ksat_type == 'recip':
        try:
            ks = df_params['ksurface']
            d = df_params['kdecay']

            ksat = bind_avg_recip_ksat(ks, d)
            print('using recip ksat')

        except KeyError:
            print('could not find parameters ksurface and/or kdecay for ksat_type %s'%ksat_type)

    elif ksat_type == 'exp':
        try:
            ks = df_params['ksurface']
            k0  = df_params['kdepth']
            dk = df_params['kdecay']

            ksat = bind_avg_exp_ksat(ks, k0, dk)
        except KeyError:
            print('could not find parameters ksurface, kdepth, and/or kdecay for ksat_type %s'%ksat_type)
    else:
        print('Could not find ksat_type %s'%ksat_type)
        raise KeyError
except KeyError:
    ksat = df_params['ksat']

#initialize grid
mg = RasterModelGrid(grid.shape, xy_spacing=grid.dx)
bc_dict = {'4':mg.BC_NODE_IS_CLOSED, '1':mg.BC_NODE_IS_FIXED_VALUE}
if bc is not None:
    mg.set_status_at_node_on_edges(
            right=bc_dict[bc[0]],
            top=bc_dict[bc[1]],
            left=bc_dict[bc[2]],
            bottom=bc_dict[bc[3]],
    )       
else:
    mg.set_status_at_node_on_edges(
            right=mg.BC_NODE_IS_CLOSED,
            top=mg.BC_NODE_IS_CLOSED,
            left=mg.BC_NODE_IS_FIXED_VALUE,
            bottom=mg.BC_NODE_IS_CLOSED,
    )

#%%
# initialize grid and components

#initialize grid
z = mg.add_zeros('node', 'topographic__elevation')
z[:] = elev
z[np.isnan(z)] = b
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = base
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = base + wtrel * b

# f = open('../post_proc/%s/Q_pts_%d.csv'%(base_output_path, ID), 'w')
# def write_Q(grid, r, dt, file=f):
#     cores = grid.core_nodes
#     h = grid.at_node["aquifer__thickness"]
#     area = grid.cell_area_at_node
#     storage = np.sum(n*h[cores]*area[cores])

#     qs = grid.at_node["surface_water__specific_discharge"]
#     qs_tot = np.sum(qs[cores]*area[cores])
#     qs_nodes = np.sum(qs[cores]>1e-10)

#     r_tot = np.sum(r[cores]*area[cores])

#     file.write('%f, %f, %f, %f, %f\n'%(dt, r_tot, qs_tot, storage, qs_nodes))


#initialize components
gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=ne,
                                  hydraulic_conductivity=ksat,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.05,
                                  vn_coefficient = 0.05,
                                  #callback_fun = write_SQ,
                                  )
pdr = PrecipitationDistribution(mg, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=T_h)
pdr.seed_generator(seedval=2)
svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                available_water_content=na,
                profile_depth=b,
                num_bins=500,
                )
svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
hm = HydrologyEventVadoseStreamPower(
                                    mg,
                                    precip_generator=pdr,
                                    groundwater_model=gdp,
                                    vadose_model=svm,
                                    )

#%%
#run model

hm.run_step()
hm.run_step_record_state()

#%% 
# #dump

with open(os.path.join(directory,base_output_path,f'Qall_{ID}.pkl'), 'wb') as file:
    pickle.dump(hm.Q_all, file)

#%% load

with open(os.path.join(directory,base_output_path,f'Qall_{ID}.pkl'), 'rb') as file:
	Q_all = pickle.load(file)



#%% 
# analysis of discharge

# indices based on average runoff from post processing
Q_mean_end_storm = grid.at_node['Q_mean_end_storm']
TI = grid.at_node['topographic__index_D8']
percs = [99,95,90]
ids = []
ps = []
vals = []

# get five points in the neighborhood of the quantiles, since places don't necessarily 
# behave similarly even when close together
for p in percs:
    inds, Qs = get_neighborhood_indices(Q_mean_end_storm, p, 5, mask=grid.core_nodes)
    ids += inds
    ps += [p]*5


#%%
# calculate exceedance

# get existing full-domain discharge timeseries from csv
df = pd.read_csv(os.path.join(directory,base_output_path,f'dt_qs_s_{ID}.csv'), sep=',',header=None, names=['dt','r', 'qs', 'S', 'sat_nodes'])
# remove the first row (test row written by init of gdp)
df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)
# normalize by area
Atot = np.sum(mg.cell_area_at_node[mg.core_nodes])
df['qs_norm'] = df['qs']/Atot

# get some info from the hydrological model that was just run
area = grid.at_node['drainage_area']
Q_all = hm.Q_all[1:,:]
dt = np.diff(hm.time)
intensity = hm.intensity[:-1]

# sort and calculate exceedance for Q at ids
Q_sorted_ids = np.zeros((Q_all.shape[0], len(ids)))
dt_sorted_ids = np.zeros_like(Q_sorted_ids)
exceedance_ids = np.zeros_like(Q_sorted_ids)
for i, id in enumerate(ids):
    Q = Q_all[:,id]/area[id] * 3600 * 1000
    sorted_inds = np.flip(np.argsort(Q))
    Q_sorted_ids[:,i] = Q[sorted_inds]
    dt_sorted_ids[:,i] = dt[sorted_inds]
    exceedance_ids[:,i] = np.cumsum(dt_sorted_ids[:,i])/np.sum(dt)

# sort and calculate exceedance for rainfall 
sorted_inds = np.flip(np.argsort(intensity))
sorted_i = intensity[sorted_inds]
dt_sorted_i = dt[sorted_inds]
exceedance_i = np.cumsum(dt_sorted_i)/np.sum(dt)

# sort and calculate exceedance for Qtot from the csv
Qtot = df['qs_norm'].values
dt_Qtot = df['dt'].values
sorted_inds = np.flip(np.argsort(Qtot))
sorted_Qtot = Qtot[sorted_inds]
dt_sorted_Qtot = dt_Qtot[sorted_inds]
exceedance_Qtot = np.cumsum(dt_sorted_Qtot)/np.sum(dt_Qtot)


#%% 
# plot exceedance

viridis = cm.get_cmap('viridis_r', len(percs))
vir = viridis.colors

c_class= 5*[0]+5*[1]+5*[2]
fig, ax = plt.subplots()
for i, id in enumerate(ids):
    ax.scatter(Q_sorted_ids[:,i], exceedance_ids[:,i], s=2, c=vir[c_class[i]], label=r"~$Q_{%d}$"%ps[i])
ax.scatter(sorted_i*1e3*3600, exceedance_i, s=2, c='r', label='Rainfall')
# ax.scatter(sorted_Qtot*1e3*3600, exceedance_Qtot, s=2, c='k', label='Q total')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Q [mm/hr]')
ax.set_ylabel('Exceedance Frequency')
ax.set_xlim((1e-3,1e2))
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
    if l not in lablout:
        lablout.append(l)
        handout.append(h)
ax.legend(handout, lablout, loc='lower left')
plt.savefig(os.path.join(directory,base_output_path,f'Q_exceedance_{ID}.pdf'), dpi=300)



#%% Attempt at manual selection of watershed points

# transect across a larger stream channel
pts = [4508, 5518,5923,5563]

area = grid.at_node['drainage_area']
TI = grid.at_node['topographic__index_D8']
quants = [99.99,99.9,99,90]

# inds = [np.argwhere(TI==np.percentile(TI[grid.core_nodes],q,method='lower'))[0][0] for q in quants]
inds = [np.argwhere(area==np.percentile(area[grid.core_nodes],q,method='lower'))[0][0] for q in quants]

x_inds = grid.x_of_node[inds]
y_inds = grid.y_of_node[inds]
area_inds = area[inds]

fig, ax = plt.subplots()
imshow_grid(grid, 'Q_mean_end_storm', cmap='viridis')
ax.scatter(x_inds,y_inds, c=area_inds, cmap='plasma')

#%%

fig, ax = plt.subplots()
for pt in inds:
    plt.plot(hm.time/(3600*24), hm.Q_all[:,pt]/area[pt]*(3600*1e3), label=str(pt))
ax.set_yscale('log') 
plt.legend()

