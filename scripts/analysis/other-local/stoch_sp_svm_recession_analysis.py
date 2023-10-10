"""
Recession analysis of full grid for stochastic model runs with the SchenkVadoseModel.
Save specific discharge everywhere on grid at GDP subtimesteps, route it afterward,
then calculate recession parameters from that result.
"""

#%%


import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from landlab import RasterModelGrid
from landlab.io.netcdf import to_netcdf, from_netcdf
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    FlowAccumulator,
    FlowDirectorD8,
    )
from DupuitLEM.auxiliary_models import HydrologyEventVadoseStreamPower, SchenkVadoseModel
from DupuitLEM.grid_functions import bind_avg_exp_ksat, bind_avg_recip_ksat


def calc_recession_params(q, r, dt):
    """
    Calculate the Brutsaert and Niebur recession parameters a and b:
        dQ/dt = - a Q^b
    
    parameters
    ----------
    q: array of (area-normalized) discharge
    r: array of recharge or precipitation
    dt: array of timesteps (uniform or non-uniform)

    returns
    --------
    results: full statsmodels output object
    logQ: ln(Q) used in regression
    logdQdt: ln(-dQ/dt) used in regression
    """

    # iterate through timeseries and find places without recharge where
    # discharge is also decreasing
    dQdt = []
    Q = []
    for i in range(1,len(r)):
        if r[i-1] == 0.0:
            if q[i] < q[i-1]:
                dQdt.append((q[i] - q[i-1])/dt[i-1]) # check dt[i] or dt[i-1]
                Q.append((q[i] + q[i-1])/2)
    dQdt = np.array(dQdt)
    Q = np.array(Q)
    
    # remove places that could cause issues for log
    cond = np.logical_and(-dQdt>0, Q>0)
    logdQdt = np.log(-dQdt[cond])
    logQ = np.log(Q[cond])

    # linear regression ln(-dQ/dt) = b ln(Q) + ln(a)
    x = sm.add_constant(logQ)
    model = sm.OLS(logdQdt,x)
    results = model.fit()

    return results, logQ, logdQdt

#%%

# task_id = os.environ['SLURM_ARRAY_TASK_ID']
# ID = int(task_id)
# base_output_path = os.environ['BASE_OUTPUT_FOLDER']

directory = '/Users/dlitwin/Documents/Research Data/HPC output/DupuitLEMResults/post_proc'
base_output_path = 'stoch_gam_sigma_16'
model_runs = np.arange(25)
ID=12

df_params = pd.read_csv(os.path.join(directory,base_output_path,f'params_ID_{ID}.csv'), index_col=0)[str(ID)]

grid = from_netcdf(os.path.join(directory,base_output_path,f'grid_{ID}.nc'))
elev = grid.at_node['topographic__elevation']
elev[np.isnan(elev)] = 0.0
base = grid.at_node['aquifer_base__elevation']
wtrel = grid.at_node['wtrel_mean_end_interstorm']

########## Load and basic plot
# grid_files = glob.glob('./data/*.nc')
# files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
# iteration = int(files[-1].split('_')[-1][:-3])

# try:
#     grid = from_netcdf(files[-1])
# except KeyError:
#     grid = read_netcdf(files[-1])
# elev = grid.at_node['topographic__elevation']
# base = grid.at_node['aquifer_base__elevation']
# wt = grid.at_node['water_table__elevation']


########## Run hydrological model
# load parameters and save just this ID (useful because some runs in a group have been redone with diff parameters)
# df_params = pd.read_csv('parameters.csv', index_col=0)[task_id]

# get parameter types right
for ind in df_params.index:
    try:
        df_params[ind] = float(df_params[ind])
    except ValueError:
        df_params[ind] = str(df_params[ind])

ne = df_params['ne'] #drainable porosity [-]
b = df_params['b'] #characteristic depth  [m]
p = df_params['p'] #average precipitation rate [m/s]
pet = df_params['pet']
na = df_params['na'] #plant available volumetric water content
tr = df_params['tr'] #mean storm duration [s]
tb = df_params['tb'] #mean interstorm duration [s]
ds = df_params['ds'] #mean storm depth [m]
T_h = 200*(tr+tb) #20*df_params['Th'] #total hydrological time [s]

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

#%%

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
z = mg.add_zeros('node', 'topographic__elevation')
z[:] = elev
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = base
zwt = mg.add_zeros('node', 'water_table__elevation')
# zwt[:] = wt
zwt[:] = base + wtrel * b

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

#%%
# f = open('../post_proc/%s/qs_grid_%d.csv'%(base_output_path, ID), 'w')
f = open(os.path.join(directory,base_output_path,f'qs_grid_{ID}.csv'), 'w')
def write_qs(grid, r, dt, file=f):

    qs = grid.at_node["surface_water__specific_discharge"][grid.core_nodes]
    r_save = np.mean(r[grid.core_nodes])
    out = np.concatenate(([dt, r_save], qs))

    np.savetxt(f, out, fmt='%.6e', newline=", ")
    f.write("\n")
gdp.callback_fun = write_qs

hm.run_step()
f.close()

##########  Analysis

#%%
# need to route the flow to get the discharge fields at every timestep

# load results 
df = pd.read_csv(os.path.join(directory,base_output_path,f'qs_grid_{ID}.csv'), sep=',',header=None)
# remove the first row (test row written by init of gdp)
df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)

# set up new grid
mg1 = RasterModelGrid(grid.shape, xy_spacing=grid.dx)
bc_dict = {'4':mg1.BC_NODE_IS_CLOSED, '1':mg1.BC_NODE_IS_FIXED_VALUE}
if bc is not None:
    mg1.set_status_at_node_on_edges(
            right=bc_dict[bc[0]],
            top=bc_dict[bc[1]],
            left=bc_dict[bc[2]],
            bottom=bc_dict[bc[3]],
    )       
else:
    mg1.set_status_at_node_on_edges(
            right=mg.BC_NODE_IS_CLOSED,
            top=mg.BC_NODE_IS_CLOSED,
            left=mg.BC_NODE_IS_FIXED_VALUE,
            bottom=mg.BC_NODE_IS_CLOSED,
    )
z1 = mg1.add_zeros('node', 'topographic__elevation')
z1[:] = z
qs = mg1.add_zeros('node', 'local_runoff')

# initialize components to route flow
fd = FlowDirectorD8(mg1)
fa = FlowAccumulator(
    mg1,
    surface="topographic__elevation",
    flow_director=fd,
    runoff_rate="local_runoff",
)
fd.run_one_step()

#%%

# route flow, iterate through timesteps
Q = np.zeros((len(df),len(mg1.core_nodes)))
for i in range(len(df)):
    qs[mg1.core_nodes] = df.iloc[i,2:-1].values
    fa.accumulate_flow(update_flow_director=False)
    Q[i,:] = mg1.at_node["surface_water__discharge"][mg1.core_nodes]


#%%

# recession analysis, iterate through core nodes
dt = df.iloc[:,0].values / 3600 # hr
r = df.iloc[:,1].values * 1e3 * 3600 # mm/hr
Qnorm = Q / mg.at_node["drainage_area"][mg.core_nodes] * 1e3 * 3600 # mm/hr
params = np.zeros((len(mg1.core_nodes), 5))
for j, val in enumerate(mg1.core_nodes):

    if np.sum(Qnorm[:,j]>0)/len(Qnorm[:,j]) > 0.05:
        try:
            results, logQ, logdQdt = calc_recession_params(Qnorm[:,j],r,dt)
            
            a = np.exp(results.params[0])
            b = results.params[1]
            mu = np.mean(qs)
            tau = (mu**(1-b))/a
            r_squared = results.rsquared

            params[j,:] = [a,b,mu,tau,r_squared]
        except:
            params[j,:] = np.zeros(5) * np.nan

# columns = ['a', 'b', 'mu', 'tau', 'r2']
# df_rec = pd.DataFrame(params, index=mg1.core_nodes, columns=columns)
# np.ma.masked_where(np.isnan(params[:,0]), params[:,0])

# add output to grid
rec_a = mg1.add_zeros('node', 'recession_a')
rec_a[mg1.core_nodes] = params[:,0]
rec_b = mg1.add_zeros('node', 'recession_b')
rec_b[mg1.core_nodes] = params[:,1]
rec_mu = mg1.add_zeros('node', 'recession_mu')
rec_mu[mg1.core_nodes] = params[:,2]
rec_tau = mg1.add_zeros('node', 'recession_tau')
rec_tau[mg1.core_nodes] = params[:,3]
rec_rsq = mg1.add_zeros('node', 'recession_rsq')
rec_rsq[mg1.core_nodes] = params[:,4]
#%%

rec_a[mg1.core_nodes] = params[:,0]
rec_b[mg1.core_nodes] = params[:,1]
rec_mu[mg1.core_nodes] = params[:,2]
rec_tau[mg1.core_nodes] = params[:,3]
rec_rsq[mg1.core_nodes] = params[:,4]

#%% map view recession constants


rec_b[np.isnan(rec_b)] = 0.0
rec_b[rec_b<0] = 0.0
rec_masked = np.ma.masked_equal(rec_b, 0.0)

plt.figure()
mg1.imshow(rec_masked, cmap='Blues', colorbar_label='b (-)', color_for_closed='w')
plt.savefig(os.path.join(directory,base_output_path,f'b_map_{ID}.png'))

rec_tau[np.isnan(rec_tau)] = 0.0
rec_tau[rec_tau<0] = 0.0
rec_masked_tau = np.ma.masked_equal(rec_tau, 0.0)
plt.figure()
mg1.imshow(np.log10(rec_masked_tau), cmap='Greens', colorbar_label='log10(Tau) (hr)', color_for_closed='w', vmax=10)
plt.savefig(os.path.join(directory,base_output_path,f'tau_map_{ID}.png'))


#%%
rec_rsq[np.isnan(rec_rsq)] = 0.0
rec_masked_rsq = np.ma.masked_equal(rec_rsq, 0.0)
plt.figure()
mg1.imshow(rec_masked_rsq, cmap='Reds', colorbar_label='r2 (-)', color_for_closed='w')
plt.savefig(os.path.join(directory,base_output_path,f'rsq_map_{ID}.png'))

#%% plots with area

# cond = mg1.core_nodes
cond = rec_rsq > 0.5
area = mg1.at_node['drainage_area']
plt.figure()
plt.scatter(area[cond], rec_tau[cond], s=3, alpha=0.3)
plt.ylim(1e-2,1e6)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Drainage Area')
plt.ylabel('Tau (hrs)')
plt.title('Only r2>0.5')
plt.savefig(os.path.join(directory,base_output_path,f'tau_area_{ID}_rsq.png'))

plt.figure()
plt.scatter(area[cond], rec_b[cond], s=3, alpha=0.3)
plt.ylim(0,1.5)
# plt.yscale('log')
plt.xscale('log')
plt.xlabel('Drainage Area')
plt.ylabel('b (-)')
plt.title('Only r2>0.5')
plt.savefig(os.path.join(directory,base_output_path,f'b_area_{ID}_rsq.png'))

#%% plots with other things

field = grid.at_node['topographic__index_D8']
plt.figure()
plt.scatter(field[mg1.core_nodes], rec_tau[mg1.core_nodes], s=3, alpha=0.3)
plt.ylim(1e-2,1e6)
plt.yscale('log')
plt.xscale('log')
# plt.xlabel('Drainage Area')
plt.ylabel('Tau (hrs)')
# plt.savefig(os.path.join(directory,base_output_path,f'tau_area_{ID}.png'))

plt.figure()
plt.scatter(field[mg1.core_nodes], rec_b[mg1.core_nodes], s=3, alpha=0.3)
plt.ylim(0,1.5)
# plt.yscale('log')
plt.xscale('log')
# plt.xlabel('Drainage Area')
plt.ylabel('b (-)')
# plt.savefig(os.path.join(directory,base_output_path,f'b_area_{ID}.png'))


#%%

z_plot = z1.copy()
z_plot[np.isnan(z1)] = 0
plt.figure()
mg1.imshow(z_plot)

#%%
z_plot = elev.copy()
z_plot[np.isnan(elev)] = 0
plt.figure()
mg1.imshow(z_plot)


#%%
####### save things

output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:water_table__elevation",
        "at_node:recession_a",
        "at_node:recession_b",
        "at_node:recession_mu",
        "at_node:recession_tau",
        "at_node:recession_rsq",
        'at_node:surface_water_effective__discharge',
        ]

# filename = '../post_proc/%s/grid_rec_%d.nc'%(base_output_path, ID)
filename = os.path.join(directory,base_output_path,f'grid_rec_{ID}.nc')
to_netcdf(mg, filename, include=output_fields, format="NETCDF4")
# %%
