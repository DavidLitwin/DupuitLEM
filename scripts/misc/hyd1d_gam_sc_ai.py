"""
This script is used to explore how saturation varies on a 1D parabolic hillslope,
focusing on the case where we have both nonlinear diffusion and evapotranspiration.
This allows for comparison with the 2D simulations both coupled and uncoupled.
"""

import os
import time
import numpy as np
import pickle
from itertools import product
import pandas as pd

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    )
from landlab.io.netcdf import to_netcdf

from DupuitLEM.auxiliary_models import (
    HydrologyEventVadoseStreamPower,
    SchenkVadoseModel,
    )

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

def b_fun(hg, gam, hi):
    return (hg*gam)/hi

def ksat_fun(p, hg, lg, hi):
    return (lg**2*p*hi)/hg**2

def ds_fun(hg, n, gam, sigma, hi):
    return (hg*n*gam)/(hi*sigma)

def tr_fun(hg, p, n, gam, sigma, hi, rho):
    return (hg*n*gam*rho)/(p*sigma*hi)

def tb_fun(hg, p, n, gam, sigma, hi, rho):
    return (hg*n*gam)*(1-rho)/(p*sigma*hi)

def calc_z(x, Sc, U, D):
    """Nonlinear diffusion elevation profile"""
    t1 = np.sqrt(D**2 + (2*U*x/Sc)**2)
    t2 = D*np.log((t1 + D)/(2*U/Sc))
    return -Sc**2/(2*U) * (t1 - t2)

#generate dimensioned parameters
def generate_parameters(U, lg, p, n, sc, alpha, gam, hi, lam, sigma, rho, ai):

    hg = alpha*lg
    Lh = lam*lg
    D = U/(hg/lg**2)
    b = b_fun(hg, gam, hi)
    ksat = ksat_fun(p, hg, lg, hi)
    ds = ds_fun(hg, n, gam, sigma, hi)
    tr = tr_fun(hg, p, n, gam, sigma, hi, rho)
    tb = tb_fun(hg, p, n, gam, sigma, hi, rho)
    pet = ai*p

    return D, U, hg, lg, Lh, sc, ksat, p, pet, b, ds, tr, tb, n, alpha, gam, hi, lam, sigma, rho, ai

Sc_all = np.array([0.05, 0.15, 0.3, 0.6, 1.0, 10.0])
gam_all = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
lam_all = np.array([5, 10, 20, 40])
Sc1 = np.array(list(product(Sc_all, gam_all, lam_all)))[:,0]
gam1 = np.array(list(product(Sc_all, gam_all, lam_all)))[:,1]
lam1 = np.array(list(product(Sc_all, gam_all, lam_all)))[:,2]
hi = 5.0

sigma = 32
rho = 0.03
alpha = 0.15
ai = 1.5
lg = 15 # geomorphic length scale [m]
n = 0.1 # drainable porosity [-]
p = 1.0/(365*24*3600) # steady recharge rate
U = 1e-4 # m/yr
Srange = 0.2 # range of relative saturation
Nz = 500 # number of bins in vadose model
Nt = 1000; Ny = 3; Nx = 50 # num timesteps, num y nodex, num x nodes


params = np.zeros((len(gam1),21))
for i in range(len(gam1)):
    params[i,:] = generate_parameters(U, lg, p, n, Sc1[i], alpha, gam1[i], hi, lam1[i], sigma, rho, ai)

df_params = pd.DataFrame(params,columns=['D', 'U', 'hg', 'lg', 'Lh', 'sc', 'ksat', 'p', 'pet', 'b', 'ds', 'tr', 'tb', 'n',  'alpha', 'gam', 'hi', 'lam', 'sigma', 'rho', 'ai'])
df_params['td'] = (df_params['lg']*df_params['n'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer drainage time [s]
df_params['Srange'] = Srange
df_params['beta'] = (df_params['tr']+df_params['tb'])/df_params['td']
df_params['ha'] = (df_params['p']*df_params['lg'])/(df_params['ksat']*df_params['hg']/df_params['lg']) # characteristic aquifer thickness [m]
df_params['Nx'] = Nx; df_params['Ny'] = Ny; df_params['Nt'] = Nt; df_params['Nz'] = Nz
pickle.dump(df_params, open('parameters.p','wb'))

ks = df_params['ksat'][ID]
pet = df_params['pet'][ID]
Srange = df_params['Srange'][ID]
b = df_params['b'][ID]
ds = df_params['ds'][ID]
tr = df_params['tr'][ID]
tb = df_params['tb'][ID]
Lh = df_params['Lh'][ID]
D = df_params['D'][ID]
U = df_params['U'][ID]
sc = df_params['sc'][ID]
Lh = df_params['Lh'][ID]

# initialize grid
grid = RasterModelGrid((Ny, Nx), xy_spacing=Lh/Nx)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
x = grid.x_of_node
z =  calc_z(x, sc, U, D) - calc_z(x[-1], sc, U, D)
elev[:] = np.flip(z)
base = grid.add_zeros('node', 'aquifer_base__elevation')
base[:] = elev - b
wt = grid.add_zeros('node', 'water_table__elevation')
wt[:] = elev

# initialize landlab and DupuitLEM components
gdp = GroundwaterDupuitPercolator(grid,
                                  porosity=n,
                                  hydraulic_conductivity=ks,
                                  recharge_rate=0.0,
                                  vn_coefficient=0.5,
                                  courant_coefficient=0.5,
                                  )
pdr = PrecipitationDistribution(grid,
                               mean_storm_duration=tr,
                               mean_interstorm_duration=tb,
                               mean_storm_depth=ds,
                               total_t=Nt*(tr+tb))
pdr.seed_generator(seedval=1235)
svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                 available_relative_saturation=Srange,
                 profile_depth=b,
                 porosity=n,
                 num_bins=Nz,
                 )
hm = HydrologyEventVadoseStreamPower(
                                    grid,
                                    precip_generator=pdr,
                                    groundwater_model=gdp,
                                    vadose_model=svm,
                                    )


# run once to spin up model
t1 = time.time()
hm.run_step()
t2 = time.time()

# open file and make function for saving gdp subtimestep data
# f = open('./gdp_flux_state_%d.csv'%ID, 'w')
# def write_SQ(grid, r, dt, file=f):
#     cores = grid.core_nodes
#     h = grid.at_node["aquifer__thickness"]
#     area = grid.cell_area_at_node
#     storage = np.sum(n*h[cores]*area[cores])
#
#     qs = grid.at_node["surface_water__specific_discharge"]
#     qs_tot = np.sum(qs[cores]*area[cores])
#     qs_nodes = np.sum(qs[cores]>1e-10)
#
#     r_tot = np.sum(r[cores]*area[cores])
#
#     file.write('%f, %f, %f, %f, %f\n'%(dt, r_tot, qs_tot, storage, qs_nodes))
# gdp.callback_fun = write_SQ

# run and record state
hm.run_step_record_state()
# f.close()

############ Analysis ############
df_output = {}

######## Runoff generation
# find times with rain. Note in df qs and S are at the end of the timestep.
# i is at the beginning of the timestep. Assumes timeseries starts with rain.
# df = pd.read_csv('./gdp_flux_state_%d.csv'%ID, sep=',',header=None, names=['dt','r', 'qs', 'S', 'qs_nodes'])
# df['t'] = np.cumsum(df['dt'])
# df_output['qs_tot'] = np.trapz(df['qs'], df['t'])
# df_output['r_tot'] = np.sum(df['dt'] * df['r'])

# effective Qstar
Q_all = hm.Q_all[1:,:]
dt = np.diff(hm.time)
intensity = hm.intensity[:-1]
qstar_mean = grid.add_zeros('node', 'qstar_mean_no_interevent')

# mean Q based on the geomorphic definition - only Q during storm events does geomorphic work
Q_event_sum = np.zeros(Q_all.shape[1])
for i in range(1,len(Q_all)):
    if intensity[i] > 0.0:
        Q_event_sum += 0.5*(Q_all[i,:]+Q_all[i-1,:])*dt[i]
qstar_mean[:] = (Q_event_sum/np.sum(dt[1:]))/(grid.at_node['drainage_area']*df_params['p'][ID])
qstar_mean[np.isnan(qstar_mean)] = 0.0

# mean and variance of water table
wt_all = hm.wt_all[1:,:]
base_all = np.ones(wt_all.shape)*grid.at_node['aquifer_base__elevation']
elev_all = np.ones(wt_all.shape)*grid.at_node['topographic__elevation']
wtrel_all = np.zeros(wt_all.shape)
wtrel_all[:, grid.core_nodes] = (wt_all[:, grid.core_nodes] - base_all[:, grid.core_nodes])/(elev_all[:, grid.core_nodes] - base_all[:, grid.core_nodes])

# water table and saturation at end of storm and interstorm
thresh = 1e-10 #np.mean(grid.cell_area_at_node[grid.core_nodes])*df_params['p'][ID]
sat_all = (wtrel_all > 0.99)
wtrel_end_interstorm = grid.add_zeros('node', 'wtrel_mean_end_interstorm')
wtrel_end_storm = grid.add_zeros('node', 'wtrel_mean_end_storm')
wtrel_max = grid.add_zeros('node', 'wtrel_99')
wtrel_min = grid.add_zeros('node', 'wtrel_01')
sat_end_interstorm = grid.add_zeros('node', 'sat_mean_end_interstorm')
sat_end_storm = grid.add_zeros('node', 'sat_mean_end_storm')
Q_end_interstorm = grid.add_zeros('node', 'Q_mean_end_interstorm')
Q_end_storm = grid.add_zeros('node', 'Q_mean_end_storm')

wtrel_end_storm[:] = np.mean(wtrel_all[intensity>0,:], axis=0)
wtrel_end_interstorm[:] = np.mean(wtrel_all[intensity==0.0,:], axis=0)
wtrel_max[:] = np.percentile(wtrel_all, 99, axis=0)
wtrel_min[:] = np.percentile(wtrel_all, 1, axis=0)
sat_end_storm[:] = np.mean(sat_all[intensity>0,:], axis=0)
sat_end_interstorm[:] = np.mean(sat_all[intensity==0.0,:], axis=0)
Q_end_storm[:] = np.mean(Q_all[intensity>0,:], axis=0)
Q_end_interstorm[:] = np.mean(Q_all[intensity==0.0,:], axis=0)

# classify zones of saturation
sat_class = grid.add_zeros('node', 'saturation_class')
sat_never = np.logical_and(sat_end_storm < 0.01, sat_end_interstorm < 0.01)
sat_always = np.logical_and(sat_end_interstorm > 0.99, sat_end_storm > 0.99)
sat_variable = ~np.logical_or(sat_never, sat_always)
sat_class[sat_never] = 0
sat_class[sat_variable] = 1
sat_class[sat_always] = 2
df_output['sat_never'] = np.sum(sat_never[grid.core_nodes])/grid.number_of_core_nodes
df_output['sat_variable'] = np.sum(sat_variable[grid.core_nodes])/grid.number_of_core_nodes
df_output['sat_always'] = np.sum(sat_always[grid.core_nodes])/grid.number_of_core_nodes

#### saturtion probability and entropy
calc_entropy = lambda x: -x*np.log2(x) - (1-x)*np.log2(1-x)

# first method: time weighted variability in saturation
sat_prob = grid.add_zeros('node', 'saturation_probability')
sat_entropy = grid.add_zeros('node', 'saturation_entropy')
sat_prob[:] = np.sum((sat_all.T*dt)/np.sum(dt), axis=1)
sat_entropy[:] = calc_entropy(sat_prob)
df_output['entropy_sat_variable'] = np.sum(sat_entropy[sat_variable])

# second method: interstorm-storm unsat-sat probability
sat_unsat_prob = grid.add_zeros('node', 'sat_unsat_union_probability')
# P(sat_storm and unsat_interstorm) = P(sat_storm given unsat_interstorm)*P(unsat_interstorm)
sat_storms = sat_all[intensity>0,:]; sat_interstorms = sat_all[intensity==0.0,:] # first storm record precedes interstorm record
sat_storms = sat_storms[1:,:]; sat_interstorms = sat_interstorms[:-1,:] # adjust indices so that prev interstorm is aligned with storm
p_unsat_interstorm = np.sum(~sat_interstorms, axis=0)/len(sat_interstorms) # prob unsaturated at the end of an interstorm
p_cond_sat_storm_unsat_interstorm = np.sum(sat_storms*~sat_interstorms, axis=0)/np.sum(~sat_interstorms, axis=0) # prob that saturated at end of storm given that it's unsaturated at end of interstorm
p_union_sat_storm_unsat_interstorm = p_cond_sat_storm_unsat_interstorm*p_unsat_interstorm # prob that unsaturated at end of interstorm and saturated at end of storm
sat_unsat_prob[:] = p_union_sat_storm_unsat_interstorm

df_output['runtime'] = t2 - t1

# save output
output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:wtrel_mean_end_storm",
        "at_node:wtrel_mean_end_interstorm",
        "at_node:wtrel_99",
        "at_node:wtrel_01",
        "at_node:sat_mean_end_storm",
        "at_node:sat_mean_end_interstorm",
        "at_node:Q_mean_end_storm",
        "at_node:Q_mean_end_interstorm",
        "at_node:saturation_class",
        "at_node:saturation_probability",
        "at_node:saturation_entropy",
        "at_node:sat_unsat_union_probability",
        "at_node:qstar_mean_no_interevent",
        ]
to_netcdf(grid, 'grid_%d.nc'%ID, include=output_fields, format="NETCDF4")

pickle.dump(df_output, open('output_%d.p'%ID, 'wb'))
pickle.dump(grid, open('grid_%d.p'%ID, 'wb'))
