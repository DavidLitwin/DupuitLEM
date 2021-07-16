"""
Analysis of results on HPC for stochastic stream power model runs.

update for new stochastic models
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from landlab import imshow_grid, RasterModelGrid, HexModelGrid, LinkStatus
from landlab.io.netcdf import to_netcdf, from_netcdf, read_netcdf
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    HeightAboveDrainageCalculator,
    DrainageDensity,
    )
from landlab.grid.mappers import map_downwind_node_link_max_to_node
from DupuitLEM.auxiliary_models import HydrologyEventVadoseStreamPower, SchenkVadoseModel

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']

########## Load and basic plot
grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
iteration = int(files[-1].split('_')[-1][:-3])

try:
    grid = from_netcdf(files[-1])
except KeyError:
    grid = read_netcdf(files[-1])
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt = grid.at_node['water_table__elevation']

# elevation
plt.figure(figsize=(8,6))
imshow_grid(grid, elev, cmap='gist_earth', colorbar_label='Elevation [m]', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID,iteration))
plt.savefig('../post_proc/%s/elev_ID_%d.png'%(base_output_path, ID))
plt.close()


########## Run hydrological model
# load parameters and save just this ID (useful because some runs in a group have been redone with diff parameters)
df_params = pickle.load(open('./parameters.p','rb'))
params = df_params.iloc[ID]
pickle.dump(params, open('../post_proc/%s/params_ID_%d.p'%(base_output_path,ID),'wb'))

Ks = df_params['ksat'][ID] #hydraulic conductivity [m/s]
n = df_params['n'][ID] #drainable porosity [-]
b = df_params['b'][ID] #characteristic depth  [m]
p = df_params['p'][ID] #average precipitation rate [m/s]
pet = df_params['pet'][ID]
Srange = df_params['Srange'][ID]
tr = df_params['tr'][ID] #mean storm duration [s]
tb = df_params['tb'][ID] #mean interstorm duration [s]
ds = df_params['ds'][ID] #mean storm depth [m]
T_h = 10*df_params['Th'][ID] #total hydrological time [s]

#initialize grid
dx = grid.dx
mg = RasterModelGrid(grid.shape, xy_spacing=dx)
mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_CLOSED, top=mg.BC_NODE_IS_CLOSED, \
                              left=mg.BC_NODE_IS_FIXED_VALUE, bottom=mg.BC_NODE_IS_CLOSED)
z = mg.add_zeros('node', 'topographic__elevation')
z[:] = elev
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = base
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = wt

f = open('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), 'w')
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

#initialize components
if isinstance(mg, RasterModelGrid):
    method = 'D8'
elif isinstance(mg, HexModelGrid):
    method = 'Steepest'
else:
    raise TypeError("grid should be Raster or Hex")

gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=n,
                                  hydraulic_conductivity=Ks,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.05,
                                  vn_coefficient = 0.05,
                                  callback_fun = write_SQ,
                                  )
pdr = PrecipitationDistribution(mg, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=T_h)
pdr.seed_generator(seedval=2)
svm = SchenkVadoseModel(
                potential_evapotranspiration_rate=pet,
                 available_relative_saturation=Srange,
                 profile_depth=b,
                 porosity=n,
                 num_bins=500,
                 )
hm = HydrologyEventVadoseStreamPower(
                                    grid,
                                    precip_generator=pdr,
                                    groundwater_model=gdp,
                                    vadose_model=svm,
                                    )

#run model
hm.run_step_record_state()
f.close()

##########  Analysis

#dataframe for output
df_output = {}

#load the full storage discharge dataset that was just generated
df = pd.read_csv('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), sep=',',header=None, names=['dt','i', 'qs', 'S', 'qs_cells'])
# remove the first row (test row written by init of gdp)
df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)
# make dimensionless timeseries
Atot = np.sum(mg.cell_area_at_node[mg.core_nodes])
df['qs_star'] = df['qs']/(p*Atot)
df['S_star'] = df['S']/(b*n*Atot)

##### recession
def power_law(x, a, b, c):
    return a*np.power(x, b) + c

def linear_law(x,a,c):
    return a*x + c

# find recession periods
rec_inds = np.where(np.diff(df['qs_star'], prepend=0.0) < 0.0)[0]
Qrec = df['qs_star'][rec_inds]
Srec = df['S_star'][rec_inds]

# try to fit linear and power fits to Q-S relationship directly
try:
    pars, cov = curve_fit(f=power_law, xdata=Srec, ydata=Qrec, p0=[0, 1, 0], bounds=(-100, 100))
    stdevs = np.sqrt(np.diag(cov))

    pars_lin, cov_lin = curve_fit(f=linear_law, xdata=Srec, ydata=Qrec, p0=[0, 0], bounds=(-100, 100))
    stdevs_lin = np.sqrt(np.diag(cov_lin))

    df_output['rec_a'] = pars[0]
    df_output['rec_b'] = pars[1]
    df_output['rec_c'] = pars[2]
    df_output['rec_a_std'] = stdevs[0]
    df_output['rec_b_std'] = stdevs[1]
    df_output['rec_a_linear'] = pars_lin[0]
    df_output['rec_c_linear'] = pars_lin[1]
    df_output['rec_a_std_linear'] = stdevs_lin[0]

except:
    print("error fitting recession")

##### steepness, curvature, and topographic index
S8 = mg.add_zeros('node', 'slope_D8')
S4 = mg.add_zeros('node', 'slope_D4')
curvature = mg.add_zeros('node', 'curvature')
steepness = mg.add_zeros('node', 'steepness')
TI8 = mg.add_zeros('node', 'topographic__index_D8')
TI4 = mg.add_zeros('node', 'topographic__index_D4')

#slope for steepness is the absolute value of D8 gradient associated with
#flow direction. Same as FastscapeEroder. curvature is divergence of gradient.
#Same as LinearDiffuser. TI is done both ways.
dzdx_D8 = mg.calc_grad_at_d8(elev)
dzdx_D4 = mg.calc_grad_at_link(elev)
dzdx_D4[mg.status_at_link == LinkStatus.INACTIVE] = 0.0
S8[:] = abs(dzdx_D8[mg.at_node['flow__link_to_receiver_node']])
S4[:] = map_downwind_node_link_max_to_node(mg, dzdx_D4)

curvature[:] = mg.calc_flux_div_at_node(dzdx_D4)
steepness[:] = np.sqrt(mg.at_node['drainage_area'])*S8
TI8[:] = mg.at_node['drainage_area']/(S8*mg.dx)
TI4[:] = mg.at_node['drainage_area']/(S4*mg.dx)

######## Runoff generation
# find times with rain. Note in df qs and S are at the end of the timestep.
# i is at the beginning of the timestep. Assumes timeseries starts with rain.
df['t'] = np.cumsum(df['dt'])
is_rain = df['i'] > 0.0
pre_rain = np.where(np.diff(is_rain*1)>0.0)[0] #last data point before rain
end_rain = np.where(np.diff(is_rain*1)<0.0)[0][1:] #last data point where there is rain

# make a linear-type baseflow separation, where baseflow during rain increases
# linearly from its initial qs before rain to its final qs after rain.
qb = df['qs'].copy()
q = df['qs']
t = df['t']
for i in range(len(pre_rain)-1): # added -1 but FIX THIS.
    slope = (q[end_rain[i]+1] - q[pre_rain[i]])/(t[end_rain[i]+1]-t[pre_rain[i]])
    qb[pre_rain[i]+1:end_rain[i]+1] = q[pre_rain[i]]+slope*(t[pre_rain[i]+1:end_rain[i]+1] - t[pre_rain[i]])

df['qb'] = qb
qe = df['qs'] - df['qb']

# integrate to find total values. trapezoidal for the properties that
# change dynamically, and simple rectangular for i bc we know rain varies in this way.
qe_tot = np.trapz(qe, df['t'])
qb_tot = np.trapz(qb, df['t'])
qs_tot = np.trapz(df['qs'], df['t'])
p_tot = np.sum(df['dt'] * df['i'])

df_output['BFI'] = qb_tot/qs_tot #baseflow index
df_output['RR'] = qe_tot/p_tot #runoff ratio

###### spatial runoff related quantities

# effective Qstar
Q_all = hm.Q_all[1:,:]
dt = np.diff(hm.time)
intensity = hm.intensity[:-1]
qstar_mean = mg.add_zeros('node', 'qstar_mean_no_interevent')

# recharge
recharge = hm.r_all[1:,:]
recharge_event = mg.add_zeros('node', 'recharge_rate_mean_storm')
recharge_event[:] = np.mean(recharge[intensity>0,:], axis=0)

# mean Q based on the geomorphic definition - only Q during storm events does geomorphic work
Q_event_sum = np.zeros(Q_all.shape[1])
for i in range(1,len(Q_all)):
    if intensity[i] > 0.0:
        Q_event_sum += 0.5*(Q_all[i,:]+Q_all[i-1,:])*dt[i]
qstar_mean[:] = (Q_event_sum/np.sum(dt[1:]))/(mg.at_node['drainage_area']*df_params['p'][ID])
qstar_mean[np.isnan(qstar_mean)] = 0.0

# mean and variance of water table
wt_all = hm.wt_all[1:,:]
base_all = np.ones(wt_all.shape)*mg.at_node['aquifer_base__elevation']
elev_all = np.ones(wt_all.shape)*mg.at_node['topographic__elevation']
wtrel_all = np.zeros(wt_all.shape)
wtrel_all[:, mg.core_nodes] = (wt_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])/(elev_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])

# water table and saturation at end of storm and interstorm
thresh = 1e-10 #np.mean(mg.cell_area_at_node[grid.core_nodes])*df_params['p'][ID]
sat_all = (Q_all > thresh)
wtrel_end_interstorm = mg.add_zeros('node', 'wtrel_mean_end_interstorm')
wtrel_end_storm = mg.add_zeros('node', 'wtrel_mean_end_storm')
sat_end_interstorm = mg.add_zeros('node', 'sat_mean_end_interstorm')
sat_end_storm = mg.add_zeros('node', 'sat_mean_end_storm')
Q_end_interstorm = mg.add_zeros('node', 'Q_mean_end_interstorm')
Q_end_storm = mg.add_zeros('node', 'Q_mean_end_storm')

wtrel_end_storm[:] = np.mean(wtrel_all[intensity>0,:], axis=0)
wtrel_end_interstorm[:] = np.mean(wtrel_all[intensity==0.0,:], axis=0)
sat_end_storm[:] = np.mean(sat_all[intensity>0,:], axis=0)
sat_end_interstorm[:] = np.mean(sat_all[intensity==0.0,:], axis=0)
Q_end_storm[:] = np.mean(Q_all[intensity>0,:], axis=0)
Q_end_interstorm[:] = np.mean(Q_all[intensity==0.0,:], axis=0)

##### channel network
#find number of saturated cells
count_sat_nodes = np.sum(sat_all,axis=1)
# find median channel network at end of storm and end of interstorm
interstorm_network_id = np.where(count_sat_nodes == np.percentile(count_sat_nodes[intensity==0], 50, interpolation='nearest'))[0][0]
storm_network_id = np.where(count_sat_nodes == np.percentile(count_sat_nodes[intensity>0], 50, interpolation='nearest'))[0][0]

#set fields
storm_network = mg.add_zeros('node', 'channel_mask_storm')
interstorm_network = mg.add_zeros('node', 'channel_mask_interstorm')
storm_network[:] = sat_all[storm_network_id,:]
interstorm_network[:] = sat_all[interstorm_network_id,:]


######## Calculate HAND
hand_storm = mg.add_zeros('node', 'hand_storm')
hand_interstorm = mg.add_zeros('node', 'hand_interstorm')

hd = HeightAboveDrainageCalculator(mg, channel_mask=storm_network)
try:
    hd.run_one_step()
    hand_storm[:] = mg.at_node["height_above_drainage__elevation"].copy()
    df_output['mean_hand_storm'] = np.mean(hand_storm[mg.core_nodes])

    hd.channel_mask = interstorm_network
    hd.run_one_step()
    hand_interstorm[:] = mg.at_node["height_above_drainage__elevation"].copy()
    df_output['mean_hand_interstorm'] = np.mean(hand_interstorm[mg.core_nodes])

except:
    print('failed to calculate HAND')

######## Calculate drainage density
dd = DrainageDensity(mg, channel__mask=np.uint8(storm_network))
try:
    channel_mask = mg.at_node['channel__mask']
    df_output['dd_storm'] = dd.calculate_drainage_density()

    channel_mask[:] = np.uint8(interstorm_network)
    df_output['dd_interstorm'] = dd.calculate_drainage_density()

except:
    print('failed to calculate drainage density')

####### calculate elevation change
try:
    output_interval = df_params['output_interval'][ID]
except KeyError:
    print('output_interval not in params table. Using default.')
    output_interval = (10/(df_params['dtg']/df_params['tg'])).round().astype(int)[ID]

dt = output_interval*df_params['dtg'][ID]

z_change = np.zeros((len(files),6))
relief_change = np.zeros((len(files), 2))
try:
    grid = from_netcdf(files[0])
except KeyError:
    grid = read_netcdf(files[0])
elev0 = grid.at_node['topographic__elevation']
relief_change[0,0] = np.sum(elev0*grid.cell_area_at_node)
for i in range(1,len(files)):

    try:
        grid = from_netcdf(files[i])
    except KeyError:
        grid = read_netcdf(files[i])
    elev = grid.at_node['topographic__elevation']

    elev_diff = abs(elev-elev0)
    z_change[i,0] = np.max(elev_diff)
    z_change[i,1] = np.percentile(elev_diff,90)
    z_change[i,2] = np.percentile(elev_diff,50)
    z_change[i,3] = np.percentile(elev_diff,10)
    z_change[i,4] = np.min(elev_diff)
    z_change[i,5] = np.mean(elev_diff)

    relief_change[i,0] = np.mean(elev[grid.core_nodes])
    relief_change[i,1] = (relief_change[i,0]- relief_change[i-1,0])/dt

    elev0 = elev.copy()

df_z_change = pd.DataFrame(z_change,columns=['max', '90 perc', '50 perc', '10 perc', 'min', 'mean'])
r_change = pd.DataFrame()
r_change['r_nd'] = relief_change[:,0]/df_params['hg'][ID]
r_change['drdt_nd'] = relief_change[:,1]*(df_params['tg'][ID]/df_params['hg'][ID])
r_change['t_nd'] = np.arange(len(files))*(dt/df_params['tg'][ID])


####### save things

output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        'at_node:channel_mask_storm',
        'at_node:channel_mask_interstorm',
        'at_node:hand_storm',
        'at_node:hand_interstorm',
        'at_node:topographic__index_D8',
        'at_node:topographic__index_D4',
        'at_node:slope_D8',
        'at_node:slope_D4',
        'at_node:drainage_area',
        'at_node:curvature',
        'at_node:steepness',
        'at_node:qstar_mean_no_interevent',
        'at_node:recharge_rate_mean_storm',
        'at_node:wtrel_mean_end_storm',
        'at_node:wtrel_mean_end_interstorm',
        'at_node:sat_mean_end_storm',
        'at_node:sat_mean_end_interstorm',
        'at_node:Q_mean_end_storm',
        'at_node:Q_mean_end_interstorm',
        ]

filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
to_netcdf(mg, filename, include=output_fields, format="NETCDF4")

pickle.dump(df_output, open('../post_proc/%s/output_ID_%d.p'%(base_output_path, ID), 'wb'))
pickle.dump(df, open('../post_proc/%s/q_s_dt_ID_%d.p'%(base_output_path, ID), 'wb'))
df_z_change.to_csv('../post_proc/%s/z_change_%d.csv'%(base_output_path, ID))
r_change.to_csv('../post_proc/%s/relief_change_%d.csv'%(base_output_path, ID))
