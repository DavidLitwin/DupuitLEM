"""
This script is used to explore how saturation varies across the space
of dimensionless parameters on a 1D parabolic hillslope. This explores
the idea presented by Freeze (1980) that only a narrow combination of slopes
and hydraulic conductivities produce variable saturation on hillslopes.
Dimensionless parameters are the same as the 2D hydrological model used in
the LEM, but now there is an additional free parameter, the hillslope length.

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
    HydrologyEventStreamPower,
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

#generate dimensioned parameters
def generate_parameters(alpha, gam, hi, sigma, rho, lam, p, lg, n):

    hg = alpha*lg
    b = b_fun(hg, gam, hi)
    ks = ksat_fun(p, hg, lg, hi)
    ds = ds_fun(hg, n, gam, sigma, hi)
    tr = tr_fun(hg, p, n, gam, sigma, hi, rho)
    tb = tb_fun(hg, p, n, gam, sigma, hi, rho)
    Lh = lam*lg

    return alpha, gam, hi, sigma, rho, lam, p, ks, hg, lg, b, ds, tr, tb, Lh


alpha_all = np.geomspace(0.05,0.5,5) # characteristic gradient
Hi = 5.0 # hillslope number
gam_all = np.geomspace(0.2,20,5) # drainage capacity
sigma = 16 # storage variability
rho = 0.03 # rainfall steadiness
lam_all = np.linspace(5,200,5)

lg = 15 # horizontal length scale
p = 1.0/(3600*24*365) # mean rainfall rate (m/s)
n = 0.1 # porosity

alpha1 = np.array(list(product(alpha_all, lam_all, gam_all)))[:,0]
lam1 = np.array(list(product(alpha_all, lam_all, gam_all)))[:,1]
gam1 = np.array(list(product(alpha_all, lam_all, gam_all)))[:,2]

Nt = 2000; Ny = 3; Nx = 50 # num timesteps, num y nodex, num x nodes

# assemble parameters dataframe
params = np.zeros((len(alpha1),15))
for i in range(len(alpha1)):
    params[i,:] = generate_parameters(alpha1[i], gam1[i], Hi, sigma, rho, lam1[i], p, lg, n)
df_params = pd.DataFrame(params,columns=['alpha', 'gam', 'hi', 'sigma', 'rho', 'lam', 'p', 'ks', 'hg', 'lg', 'b', 'ds','tr', 'tb', 'Lh'])
df_params['Nx'] = Nx; df_params['Ny'] = Ny; df_params['Nt'] = Nt
pickle.dump(df_params, open('parameters.p','wb'))

ks = df_params['ks'][ID]
b = df_params['b'][ID]
ds = df_params['ds'][ID]
tr = df_params['tr'][ID]
tb = df_params['tb'][ID]
lg = df_params['lg'][ID]
hg = df_params['hg'][ID]
Lh = df_params['Lh'][ID]

# initialize grid
grid = RasterModelGrid((Ny, Nx), xy_spacing=Lh/Nx)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
x = grid.x_of_node
# zmax = alpha*Lh
# a1 = -zmax/Lh**2; a2 = 2*zmax/Lh; a3 = b
a1 = -hg/(2*lg**2); a2 = (hg*Lh)/lg**2; a3 = b # curvature hg/lg**2 with peak at x=Lh and fixed value boundary at x=0
elev[:] = a1*x**2 + a2*x + a3
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
hm = HydrologyEventStreamPower(
                                grid,
                                precip_generator=pdr,
                                groundwater_model=gdp,
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
sat_all = (Q_all > thresh)
wtrel_end_interstorm = grid.add_zeros('node', 'wtrel_mean_end_interstorm')
wtrel_end_storm = grid.add_zeros('node', 'wtrel_mean_end_storm')
sat_end_interstorm = grid.add_zeros('node', 'sat_mean_end_interstorm')
sat_end_storm = grid.add_zeros('node', 'sat_mean_end_storm')
Q_end_interstorm = grid.add_zeros('node', 'Q_mean_end_interstorm')
Q_end_storm = grid.add_zeros('node', 'Q_mean_end_storm')

wtrel_end_storm[:] = np.mean(wtrel_all[intensity>0,:], axis=0)
wtrel_end_interstorm[:] = np.mean(wtrel_all[intensity==0.0,:], axis=0)
sat_end_storm[:] = np.mean(sat_all[intensity>0,:], axis=0)
sat_end_interstorm[:] = np.mean(sat_all[intensity==0.0,:], axis=0)
Q_end_storm[:] = np.mean(Q_all[intensity>0,:], axis=0)
Q_end_interstorm[:] = np.mean(Q_all[intensity==0.0,:], axis=0)

# classify zones of saturation
sat_class = grid.add_zeros('node', 'saturation_class')
sat_never = np.logical_and(sat_end_storm < 0.001, sat_end_interstorm < 0.001)
sat_always = np.logical_and(sat_end_interstorm > 0.999, sat_end_storm > 0.999)
sat_variable = ~np.logical_or(sat_never, sat_always)
sat_class[sat_never] = 0
sat_class[sat_variable] = 1
sat_class[sat_always] = 2
df_output['sat_never'] = np.sum(sat_never[grid.core_nodes])/grid.number_of_core_nodes
df_output['sat_variable'] = np.sum(sat_variable[grid.core_nodes])/grid.number_of_core_nodes
df_output['sat_always'] = np.sum(sat_always[grid.core_nodes])/grid.number_of_core_nodes

# saturtion probability and entropy
sat_prob = grid.add_zeros('node', 'saturation_probability')
sat_entropy = grid.add_zeros('node', 'saturation_entropy')
calc_entropy = lambda x: -x*np.log(x)
sat_prob[:] = np.sum((sat_all.T*dt)/np.sum(dt), axis=1)
sat_entropy[:] = calc_entropy(sat_prob)
df_output['entropy_sat_total'] = np.sum(sat_entropy[grid.core_nodes])
df_output['entropy_sat_variable_norm'] = np.sum(sat_entropy[sat_variable])*df_output['sat_variable']

df_output['runtime'] = t2 - t1

# save output
output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:wtrel_mean_end_storm",
        "at_node:wtrel_mean_end_interstorm",
        "at_node:wtrel_max",
        "at_node:wtrel_min",
        "at_node:sat_mean_end_storm",
        "at_node:sat_mean_end_interstorm",
        "at_node:Q_mean_end_storm",
        "at_node:Q_mean_end_interstorm",
        "at_node:saturation_class",
        "at_node:saturation_probability",
        "at_node:saturation_entropy",
        "at_node:qstar_mean_no_interevent",
        ]
to_netcdf(grid, 'grid_%d.nc'%ID, include=output_fields, format="NETCDF4")

pickle.dump(df_output, open('output_%d.p'%ID, 'wb'))
