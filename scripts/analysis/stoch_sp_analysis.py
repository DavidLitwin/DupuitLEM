"""
Analysis of results on MARCC for stochastic stream power model runs.
"""

import os
import glob
from re import sub
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from landlab import imshow_grid
from landlab.io.netcdf import read_netcdf, write_raster_netcdf

from landlab import RasterModelGrid, LinkStatus
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
T_h = 365*24*3600 #total hydrological time [s]

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
                                  courant_coefficient=0.1,
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

##### channel network
Q_all = hm.Q_all[30:,:]

#find number of saturated cells
Q_nodes = Q_all > 1e-10
count_Q_nodes = np.sum(Q_nodes,axis=1)
#find channel network with min, max, and median number of contributing cells
min_network_id = np.where(count_Q_nodes == min(count_Q_nodes))[0][0]
max_network_id = np.where(count_Q_nodes == max(count_Q_nodes))[0][0]
med_network_id = np.where(count_Q_nodes == np.median(count_Q_nodes))[0][0]

#set fields
min_network = mg.add_zeros('node', 'channel_mask_min')
max_network = mg.add_zeros('node', 'channel_mask_max')
med_network = mg.add_zeros('node', 'channel_mask_med')
min_network[:] = Q_nodes[min_network_id,:]
max_network[:] = Q_nodes[max_network_id,:]
med_network[:] = Q_nodes[med_network_id,:]

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
# is is at the beginning of the timestep.
df['t'] = np.cumsum(df['dt'])
is_rain = df['i'] > 0.0
pre_rain = np.where(np.diff(is_rain*1)>0.0)[0] #last data point before rain
end_rain = np.where(np.diff(is_rain*1)<0.0)[0][1:] #last data point where there is rain

# make a linear-type baseflow separation, where baseflow during rain increases
# linearly from its initial qs before rain to its final qs after rain.
qb = df['qs'].copy()
t = df['t']
for i in range(len(pre_rain)-1): # added -1 but FIX THIS.
    slope = (qb[end_rain[i]+1] - qb[pre_rain[i]])/(t[end_rain[i]+1]-t[pre_rain[i]])
    qb[pre_rain[i]+1:end_rain[i]+1] = qb[pre_rain[i]]+slope*(t[pre_rain[i]+1:end_rain[i]+1] - t[pre_rain[i]])

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

#quantiles of Q*
#these are maps of Q* when the max discharge is at percs percentile.
# Quantiles and the mean are time-weighted.
Q_all = hm.Q_all[1:,:]
dt = np.diff(hm.time)
Q_max = np.max(Q_all,axis=1)
percs = [90,50,10]

Q_star_percs = np.zeros((Q_all.shape[1],len(percs)))
for i in range(len(percs)):
    index = np.where(Q_max==weighted_percentile(Q_max, percs[i], weights=dt))[0][0]
    Q_star_percs[:,i] = Q_all[index,:]/(mg.at_node['drainage_area']*df_params['p'][ID])
df_qstar = pd.DataFrame(data=Q_star_percs, columns=percs)

Qmass_all = (Q_all.T * dt).T
df_qstar['mean'] = (np.sum(Qmass_all,axis=0)/np.sum(dt))/(mg.at_node['drainage_area']*df_params['p'][ID])
df_qstar['max'] = np.where(Q_max==max(Q_max))[0][0]
df_qstar['max'] = np.where(Q_max==min(Q_max))[0][0]
df_qstar.fillna(value=0,inplace=True)


######## Calculate HAND
hand_min = mg.add_zeros('node', 'hand_min')
hand_max = mg.add_zeros('node', 'hand_max')
hand_med = mg.add_zeros('node', 'hand_med')

hd = HeightAboveDrainageCalculator(mg, channel_mask=min_network)

hd.run_one_step()
hand_min[:] = mg.at_node["height_above_drainage__elevation"].copy()
df_output['mean_hand_min'] = np.mean(hand_min[mg.core_nodes])

hd.channel_mask = max_network
hd.run_one_step()
hand_max[:] = mg.at_node["height_above_drainage__elevation"].copy()
df_output['mean_hand_max'] = np.mean(hand_max[mg.core_nodes])

hd.channel_mask = med_network
hd.run_one_step()
hand_med[:] = mg.at_node["height_above_drainage__elevation"].copy()
df_output['mean_hand_med'] = np.mean(hand_med[mg.core_nodes])

######## Calculate drainage density

dd = DrainageDensity(mg, channel__mask=np.uint8(min_network))
channel_mask = mg.at_node['channel__mask']
df_output['dd_min'] = dd.calculate_drainage_density()

# channel_mask[:] = np.uint8(max_network)
# df_output['dd_max'] = dd.calculate_drainage_density()

channel_mask[:] = np.uint8(med_network)
df_output['dd_med'] = dd.calculate_drainage_density()

####### calculate topographic index
TI = mg.add_zeros('node', 'topographic__index')
S = mg.calc_slope_at_node(elev)
TI[:] = mg.at_node['drainage_area']/(S*mg.dx)

crit_twi = mg.add_zeros('node', 'TI_exceedence_contour')
twi_contour = df_params['ksat'][ID]/(df_output['rec_a_linear']*df_params['n'][ID])
crit_twi[:] = TI >= twi_contour


####### calculate elevation change
z_change = np.zeros((len(files),5))
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
    z_change[i,4] = np.mean(elev_diff)

    elev0 = elev.copy()

df_z_change = pd.DataFrame(z_change,columns=['max', '90 perc', '50 perc', '10 perc', 'mean'])
df_z_change.to_csv(path_or_buf='../post_proc/%s/z_change_%d.csv'%(base_output_path, ID))

####### save things

output_fields = [
        "topographic__elevation",
        "aquifer_base__elevation",
        'channel_mask_min',
        'channel_mask_max',
        'channel_mask_med',
        'hand_min',
        'hand_max',
        'hand_med',
        'topographic__index',
        'TI_exceedence_contour',
        'slope',
        'drainage_area',
        'curvature',
        'steepness',
        ]

filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
write_raster_netcdf(filename, mg, names = output_fields, format="NETCDF4")

pickle.dump(df_output, open('../post_proc/%s/output_ID_%d.p'%(base_output_path, ID), 'wb'))

pickle.dump(df_qstar, open('../post_proc/%s/q_star_ID_%d.p'%(base_output_path, ID), 'wb'))

pickle.dump(df, open('../post_proc/%s/q_s_dt_ID_%d.p'%(base_output_path, ID), 'wb'))
