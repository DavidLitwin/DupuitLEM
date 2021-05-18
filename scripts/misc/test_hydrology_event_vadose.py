
import os
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

def ds_fun(hg, n, beta, hi):
    return (hg*n*beta)/hi

def tr_fun(hg, p, n, beta, hi, rho):
    return (hg*n*beta*rho)/(p*hi)

def tb_fun(hg, p, n, beta, hi, rho):
    return (hg*n*beta)*(1-rho)/(p*hi)

#generate dimensioned parameters
def generate_parameters(alpha, gam, hi, beta, rho, ai, p, lg, n):

    hg = alpha*lg
    b = b_fun(hg, gamma, hi)
    ks = ksat_fun(p, hg, lg, hi)
    d = ds_fun(hg, n, beta, hi)
    tr = tr_fun(hg, p, n, beta, hi, rho)
    tb = tb_fun(hg, p, n, beta, hi, rho)
    pet = ai*p

    return alpha, gam, hi, beta, rho, ai, p, pet, ks, hg, lg, b, d, tr, tb


lg = 15 # horizontal length scale
alpha = 0.1 # characteristic gradient
Hi = 10.0 # hillslope number
gamma = 1.0 # drainage capacity
beta_all = [0.01, 0.5, 1.0, 2.0] # storage variability
rho = 0.01 # rainfall steadiness
Ai_all = [0.1, 1.0, 4.0] # aridity
p = 1.0/(3600*24*365) # mean rainfall rate (m/s)
n = 0.1 # porosity
Smin = 0.0
Smax = 1.0
initial_wt_all = [0.1, 0.5, 1.0]

beta1 = np.array(list(product(beta_all, Ai_all, initial_wt_all)))[:,0]
Ai1 = np.array(list(product(beta_all, Ai_all, initial_wt_all)))[:,1]
initial_wt1 = np.array(list(product(beta_all, Ai_all, initial_wt_all)))[:,2]

Nt = 5000; Nx = 50; Ny = 3; Nz = 500
xmax = 10*lg

# assemble parameters dataframe
params = np.zeros((len(beta1),15))
for i in range(len(Ai1)):
    params[i,:] = generate_parameters(alpha, gamma, Hi, beta1[i], rho, Ai1[i], p, lg, n)
df_params = pd.DataFrame(params,columns=['alpha', 'gam', 'hi', 'beta', 'rho', 'ai', 'p', 'pet', 'ks', 'hg', 'lg', 'b', 'd','tr', 'tb'])
df_params['Lx'] = xmax
df_params['Nx'] = Nx; df_params['Ny'] = Ny; df_params['Nz'] = Nz; df_params['Nt'] = Nt
df_params['Smin'] = Smin; df_params['Smax'] = Smax
df_params['initial_wtrel'] = initial_wt1
pickle.dump(df_params, open('parameters.p','wb'))

pet = df_params['pet'][ID]
ks = df_params['ks'][ID]
b = df_params['b'][ID]
d = df_params['d'][ID]
tr = df_params['tr'][ID]
tb = df_params['tb'][ID]
wtrel0 = df_params['initial_wtrel'][ID]


grid = RasterModelGrid((Nx, Ny), xy_spacing=xmax/Nx)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
elev = grid.add_zeros('node', 'topographic__elevation')
x = grid.x_of_node
ymax = alpha*xmax
a1 = -ymax/xmax**2; a2 = 2*ymax/xmax; a3 = b
elev[:] = a1*x**2 + a2*x + a3
base = grid.add_zeros('node', 'aquifer_base__elevation')
base[:] = elev - b
wt = grid.add_zeros('node', 'water_table__elevation')
wt[grid.core_nodes] = base[grid.core_nodes] + wtrel0*b
wt[grid.open_boundary_nodes] = elev[grid.open_boundary_nodes]

f = open('./gdp_flux_state_%d.csv'%ID, 'w')
def write_SQ(grid, r, dt, file=f):
    cores = grid.core_nodes
    h = grid.at_node["aquifer__thickness"]
    area = grid.cell_area_at_node
    storage = np.sum(n*h[cores]*area[cores])

    qs = grid.at_node["surface_water__specific_discharge"]
    qs_tot = np.sum(qs[cores]*area[cores])
    qs_nodes = np.sum(qs[cores]>1e-10)

    r_tot = np.sum(r[cores]*area[cores])

    file.write('%f, %f, %f, %f, %f\n'%(dt, r_tot, qs_tot, storage, qs_nodes))

# initialize landlab components
gdp = GroundwaterDupuitPercolator(grid,
                                  porosity=n,
                                  hydraulic_conductivity=ks,
                                  recharge_rate=0.0,
                                  vn_coefficient=0.1,
                                  courant_coefficient=0.1,
                                  callback_fun=write_SQ,
                                  )
pdr = PrecipitationDistribution(grid,
                               mean_storm_duration=tr,
                               mean_interstorm_duration=tb,
                               mean_storm_depth=d,
                               total_t=Nt*(tr+tb))
pdr.seed_generator(seedval=1235)
svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                 upper_relative_saturation=Smax,
                 lower_relative_saturation=Smin,
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


hm.run_step_record_state()
f.close()

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
qstar_mean[:] = (Q_event_sum/np.sum(dt[1:]))/(grid.at_node['drainage_area']*p)
qstar_mean[np.isnan(qstar_mean)] = 0.0

# water table
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

# precipitation and recharge recharge
mean_r_storm = grid.add_zeros('node', 'mean_recharge_storm')
mean_r_storm[:] = np.mean(hm.r_all[hm.intensity>0,:], axis=0)
p_tot = np.sum(np.array(hm.intensities)*np.array(hm.storm_dts))

#load the full storage discharge dataset that was just generated
df = pd.read_csv('./gdp_flux_state_%d.csv'%ID, sep=',',header=None, names=['dt','r', 'qs', 'S', 'qs_cells'])
# remove the first row (test row written by init of gdp)
df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)
# make dimensionless timeseries
Atot = np.sum(grid.cell_area_at_node[grid.core_nodes])
df['qs_star'] = df['qs']/(p*Atot)
df['S_star'] = df['S']/(b*n*Atot)

######## Runoff generation
# find times with recharge. Note in df qs and S are at the end of the timestep.
# i is at the beginning of the timestep. Assumes timeseries starts with recharge.
df['t'] = np.cumsum(df['dt'])
is_recharge = df['r'] > 0.0
pre_recharge = np.where(np.diff(is_recharge*1)>0.0)[0] #last data point before recharge
end_recharge = np.where(np.diff(is_recharge*1)<0.0)[0][1:] #last data point where there is recharge

# make a linear-type baseflow separation, where baseflow during recharge increases
# linearly from its initial qs before recharge to its final qs after recharge.
qb = df['qs'].copy()
q = df['qs']
t = df['t']
for i in range(len(pre_recharge)-1): # added -1 but FIX THIS.
    slope = (q[end_recharge[i]+1] - q[pre_recharge[i]])/(t[end_recharge[i]+1]-t[pre_recharge[i]])
    qb[pre_recharge[i]+1:end_recharge[i]+1] = q[pre_recharge[i]]+slope*(t[pre_recharge[i]+1:end_recharge[i]+1] - t[pre_recharge[i]])

df['qb'] = qb
qe = df['qs'] - df['qb']

df_output = {}

# integrate to find total values. trapezoidal for the properties that
# change dynamically, and simple rectangular for i bc we know recharge varies in this way.
df_output['qe_tot'] = np.trapz(qe, df['t'])
df_output['qb_tot'] = np.trapz(qb, df['t'])
df_output['qs_tot'] = np.trapz(df['qs'], df['t'])
df_output['r_tot'] = np.sum(df['dt'] * df['r']) # total recharge
df_output['p_tot'] = p_tot # this is tanek from the hm, since gdp only sees recharge

df_output['BFI'] = df_output['qb_tot']/df_output['qs_tot'] #baseflow index
df_output['RR'] = df_output['qe_tot']/df_output['p_tot'] #runoff ratio

df_output['mean recharge depth profile'] = hm.mean_recharge_depth
df_output['recharge frequency depth profile'] = hm.recharge_frequency

output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        "at_node:mean_recharge_storm"
        "at_node:qstar_mean_no_interevent",
        "at_node:wtrel_mean_end_storm",
        "at_node:wtrel_mean_end_interstorm",
        "at_node:sat_mean_end_storm",
        "at_node:sat_mean_end_interstorm",
        "at_node:Q_mean_end_storm",
        "at_node:Q_mean_end_interstorm",
        ]
to_netcdf(grid, 'grid_%d.nc'%ID, include=output_fields, format="NETCDF4")

pickle.dump(df_output, open('output_%d.p'%ID, 'wb'))
pickle.dump(df, open('gdp_flux_state_%d.p'%ID, 'wb'))
