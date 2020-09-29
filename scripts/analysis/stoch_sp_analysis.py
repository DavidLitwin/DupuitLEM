"""
Analysis of results on HPC for stochastic stream power model runs.
"""

import os
import glob
import pickle
from re import sub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from statsmodels.stats.weightstats import DescrStatsW

from landlab import imshow_grid, RasterModelGrid, LinkStatus
from landlab.io.netcdf import read_netcdf, write_raster_netcdf
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    HeightAboveDrainageCalculator,
    DrainageDensity,
    )
from DupuitLEM.auxiliary_models import HydrologyEventStreamPower

def weighted_percentile(data, percents, weights=None, interpolation="nearest"):
    ''' percents in units of 1%
        weights specifies the frequency (count) of data.

        This is a generalization of the solution provided by
        Kambrian and Max Ghenis here:
        https://stackoverflow.com/a/31539746
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()*100
    f=interp1d(p, d, kind=interpolation)
    return f(percents)

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']

########## Load and basic plot
grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:float(sub("\D", "", x[25:-3])))

path = files[-1]
iteration = int(sub("\D", "", path[25:-3]))

grid = read_netcdf(path)
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt = grid.at_node['water_table__elevation']

# elevation
plt.figure(figsize=(8,6))
imshow_grid(grid,elev, cmap='gist_earth', colorbar_label = 'Elevation [m]', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID,iteration))
plt.savefig('../post_proc/%s/elev_ID_%d.png'%(base_output_path, ID))
plt.close()


########## Run hydrological model
df_params = pickle.load(open('./parameters.p','rb'))
df_params['hg'] = df_params['U']/df_params['K']
df_params['lg'] = np.sqrt(df_params['D']/df_params['K'])
df_params['tg'] = 1/df_params['K']
pickle.dump(df_params, open('../post_proc/%s/parameters.p'%base_output_path,'wb'))

Ks = df_params['ksat'][ID] #hydraulic conductivity [m/s]
n = df_params['n'][ID] #drainable porosity [-]
b = df_params['b'][ID] #characteristic depth  [m]
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

gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=n,
                                  hydraulic_conductivity=Ks,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.1*Ks/1e-5,
                                  vn_coefficient = 0.1*Ks/1e-5,
                                  callback_fun = write_SQ,
                                  )

pdr = PrecipitationDistribution(mg, mean_storm_duration=tr,
    mean_interstorm_duration=tb, mean_storm_depth=ds,
    total_t=T_h)
pdr.seed_generator(seedval=2)

hm = HydrologyEventStreamPower(
        mg,
        precip_generator=pdr,
        groundwater_model=gdp,
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

##### recession
def power_law(x, a, b, c):
    return a*np.power(x, b) + c

def linear_law(x,a,c):
    return a*x + c

# find recession periods
rec_inds = np.where(np.diff(df['qs'], prepend=0.0) < 0.0)[0]
Q = df['qs'][rec_inds]
S = df['S'][rec_inds] - min(df['S'][rec_inds])

# try to fit linear and power fits to Q-S relationship directly
try:
    pars, cov = curve_fit(f=power_law, xdata=S, ydata=Q, p0=[0, 1, 0], bounds=(-100, 100))
    stdevs = np.sqrt(np.diag(cov))

    pars_lin, cov_lin = curve_fit(f=linear_law, xdata=S, ydata=Q, p0=[0, 0], bounds=(-100, 100))
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
    df_output['rec_a_linear'] = 0.0

##### steepness and curvature
S = mg.add_zeros('node', 'slope')
A = mg.at_node['drainage_area']
curvature = mg.add_zeros('node', 'curvature')
steepness = mg.add_zeros('node', 'steepness')

#slope is the absolute value of D8 gradient associated with flow direction. Same as FastscapeEroder.
#curvature is divergence of gradient. Same as LinearDiffuser.
dzdx_D8 = mg.calc_grad_at_d8(elev)
dzdx_D4 = mg.calc_grad_at_link(elev)
dzdx_D4[mg.status_at_link == LinkStatus.INACTIVE] = 0.0
S[:] = abs(dzdx_D8[mg.at_node['flow__link_to_receiver_node']])

curvature[:] = mg.calc_flux_div_at_node(dzdx_D4)
steepness[:] = np.sqrt(A)*S

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
# change dynamically, and simple rectangular bc we know rain varies in this way.
qe_tot = np.trapz(qe,df['t'])
qb_tot = np.trapz(qb,df['t'])
qs_tot = np.trapz(df['qs'],df['t'])
p_tot = np.sum(df['dt']*df['i'])

df_output['BFI'] = qb_tot/qs_tot #baseflow index
df_output['RR'] = qe_tot/p_tot #runoff ratio

###### spatial runoff related quantities

# effective Qstar
Q_all = hm.Q_all[1:,:]
dt = np.diff(hm.time)
intensity = hm.intensity[:-1]
qstar_mean = mg.add_zeros('node', 'qstar_mean_no_interevent')

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
wtrel_mean = mg.add_zeros('node', 'wtrel_mean')
wtrel_std = mg.add_zeros('node', 'wtrel_std')

for i in range(len(wtrel_mean)):
    ws = DescrStatsW(wtrel_all[:,i], weights=dt, ddof=0)
    wtrel_mean[i] = ws.mean
    wtrel_std[i] = ws.std

# water table and saturation at end of storm and interstorm
sat_all = (Q_all > 1e-10)
wtrel_end_interstorm = mg.add_zeros('node', 'wtrel_mean_end_interstorm')
wtrel_end_storm = mg.add_zeros('node', 'wtrel_mean_end_storm')
sat_end_interstorm = mg.add_zeros('node', 'sat_mean_end_interstorm')
sat_end_storm = mg.add_zeros('node', 'sat_mean_end_storm')

wtrel_end_storm[:] = np.mean(wtrel_all[intensity>0,:], axis=0)
wtrel_end_interstorm[:] = np.mean(wtrel_all[intensity==0.0,:], axis=0)
sat_end_storm[:] = np.mean(sat_all[intensity>0,:], axis=0)
sat_end_interstorm[:] = np.mean(sat_all[intensity==0.0,:], axis=0)

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

####### calculate topographic index
TI = mg.add_zeros('node', 'topographic__index')
S = mg.calc_slope_at_node(elev)
TI[:] = mg.at_node['drainage_area']/(S*mg.dx)

####### calculate elevation change
z_change = np.zeros((len(files),6))
grid = read_netcdf(files[0])
elev0 = grid.at_node['topographic__elevation']
for i in range(1,len(files)):

    grid = read_netcdf(files[i])
    elev = grid.at_node['topographic__elevation']

    elev_diff = abs(elev-elev0)
    z_change[i,0] = np.max(elev_diff)
    z_change[i,1] = np.percentile(elev_diff,90)
    z_change[i,2] = np.percentile(elev_diff,50)
    z_change[i,3] = np.percentile(elev_diff,10)
    z_change[i,4] = np.min(elev_diff)
    z_change[i,5] = np.mean(elev_diff)

    elev0 = elev.copy()

df_z_change = pd.DataFrame(z_change,columns=['max', '90 perc', '50 perc', '10 perc', 'min', 'mean'])

####### save things

output_fields = [
        "topographic__elevation",
        "aquifer_base__elevation",
        'channel_mask_storm',
        'channel_mask_interstorm',
        'hand_storm',
        'hand_interstorm',
        'topographic__index',
        'slope',
        'drainage_area',
        'curvature',
        'steepness',
        'wtrel_mean',
        'wtrel_std',
        'qstar_mean_no_interevent',
        'wtrel_mean_end_interstorm',
        'wtrel_mean_end_storm',
        'sat_mean_end_interstorm',
        'sat_mean_end_storm',
        ]

filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
write_raster_netcdf(filename, mg, names = output_fields, format="NETCDF4")

pickle.dump(df_z_change, open('../post_proc/%s/z_change_%d.p'%(base_output_path, ID), 'wb'))
pickle.dump(df_output, open('../post_proc/%s/output_ID_%d.p'%(base_output_path, ID), 'wb'))
pickle.dump(df, open('../post_proc/%s/q_s_dt_ID_%d.p'%(base_output_path, ID), 'wb'))
