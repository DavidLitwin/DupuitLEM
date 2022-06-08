"""
Analysis of results on HPC for stochastic stream power model runs.

Analysis should be the same as stoch_sp_svm_analysis.py, but rather than
always analyzing the last available saved output, can specify a set of
iterations to analyze for a single model_run. Use run_batch_rock_multi.sh
to send this script.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import richdem as rd
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

base_output_path = os.environ['BASE_OUTPUT_FOLDER']
ID = int(os.environ['MODEL_RUN'])
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

########## Load and basic plot
grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
all_iterations = [int(file.split('_')[-1][:-3]) for file in files]

####### calculate relief
relief = np.zeros(len(files))
for i in range(len(files)):
    grid = from_netcdf(files[i])
    elev = grid.at_node['topographic__elevation']
    relief[i] = np.mean(elev[grid.core_nodes])

rfinal = relief[-1] # final relief
pr = np.array([0.1, 0.3, 0.6, 0.9]) # proportion of final relief
rt = pr*rfinal # corresponding relief
inds = [np.argmin(abs(relief-r)) for r in rt] # index corresponding to that relief
iterations = all_iterations[inds]
IT = iterations[task_id]

try:
    grid = from_netcdf(files[inds[task_id]])
except KeyError:
    grid = read_netcdf(files[inds[task_id]])
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt = grid.at_node['water_table__elevation']

# elevation
plt.figure(figsize=(8,6))
imshow_grid(grid, elev, cmap='gist_earth', colorbar_label='Elevation [m]', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID, IT))
plt.savefig('../post_proc/%s/elev_ID_%d_%d.png'%(base_output_path, ID, IT))
plt.close()


########## Run hydrological model
# load parameters and save just this ID (useful because some runs in a group have been redone with diff parameters)
try:
    df_params = pd.read_csv('parameters.csv', index_col=0)[str(ID)]
except FileNotFoundError:
    df_params = pickle.load(open('./parameters.p','rb'))
    df_params = df_params.iloc[ID]

Ks = df_params['ksat'] #hydraulic conductivity [m/s]
ne = df_params['ne'] #drainable porosity [-]
b = df_params['b'] #characteristic depth  [m]
p = df_params['p'] #average precipitation rate [m/s]
tg = df_params['tg']
dtg = df_params['dtg']
hg = df_params['hg']
try:
    pet = df_params['pet']
    na = df_params['na']
    tr = df_params['tr'] #mean storm duration [s]
    tb = df_params['tb'] #mean interstorm duration [s]
    ds = df_params['ds'] #mean storm depth [m]
    T_h = 2000*(tr+tb) #20*df_params['Th'] #total hydrological time [s]
except KeyError:
    df_params_1d = pd.read_csv('df_params_1d_%d.csv'%ID, index_col=0)[str(ID)]
    pet = df_params_1d['pet']
    na = df_params['na']
    tr = df_params_1d['tr'] #mean storm duration [s]
    tb = df_params_1d['tb'] #mean interstorm duration [s]
    ds = df_params_1d['ds'] #mean storm depth [m]
    T_h = 2000*(tr+tb) #df_params_1d['Nt']*(tr+tb) #total hydrological time [s]
try:
    extraction_tol = df_params['extraction_tol']
except:
    extraction_tol = 0.0

sat_cond = 0.025 # distance from surface (units of hg) for saturation

#initialize grid
mg = RasterModelGrid(grid.shape, xy_spacing=grid.dx)
mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_CLOSED, top=mg.BC_NODE_IS_CLOSED, \
                              left=mg.BC_NODE_IS_FIXED_VALUE, bottom=mg.BC_NODE_IS_CLOSED)
z = mg.add_zeros('node', 'topographic__elevation')
z[:] = elev
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = base
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = wt

f = open('../post_proc/%s/dt_qs_s_%d_%d.csv'%(base_output_path, ID, IT), 'w')
def write_SQ(grid, r, dt, file=f):
    cores = grid.core_nodes
    h = grid.at_node["aquifer__thickness"]
    area = grid.cell_area_at_node
    storage = np.sum(ne*h[cores]*area[cores])

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
                                  porosity=ne,
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
                 available_water_content=na,
                 profile_depth=b,
                 num_bins=500,
                 )
svm.generate_state_from_analytical(ds, tb, random_seed=20220408)
if extraction_tol>0:
    svm.set_max_extraction_depth(ds, tb, threshold=extraction_tol)
hm = HydrologyEventVadoseStreamPower(
                                    mg,
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

# richdem topo analysis
z = mg.at_node['topographic__elevation']
z[mg.boundary_nodes] = np.nan
zrd = rd.rdarray(z.reshape(mg.shape), no_data=-9999)
zrd.geotransform = [0.0, mg.dx, 0.0, 0.0, 0.0, mg.dx]

profile_curvature = rd.TerrainAttribute(zrd, attrib='profile_curvature')
planform_curvature = rd.TerrainAttribute(zrd, attrib='planform_curvature')
tot_curvature = rd.TerrainAttribute(zrd, attrib='curvature')
slope = rd.TerrainAttribute(zrd, attrib='slope_riserun')

slp = mg.add_zeros('node', "slope_rd")
slp[:] = slope.reshape(z.shape)

pro = mg.add_zeros('node', "profile_curvature_rd")
pro[:] = -profile_curvature.reshape(z.shape) #flip sign of profile curv

plan = mg.add_zeros('node', "planform_curvature_rd")
plan[:] = planform_curvature.reshape(z.shape)

curv = mg.add_zeros('node', "total_curvature_rd")
curv[:] = tot_curvature.reshape(z.shape)

######## Runoff generation
df_output['cum_precip'] = hm.cum_precip
df_output['cum_recharge'] = hm.cum_recharge
df_output['cum_runoff'] = hm.cum_runoff
df_output['cum_extraction'] = hm.cum_extraction
df_output['cum_gw_export'] = hm.cum_gw_export

"""ratio of total recharge to total precipitation, averaged over space and time.
this accounts for time varying recharge with precipitation rate, unsat
storage and ET, as well as spatially variable recharge with water table depth.
"""
df_output['recharge_efficiency'] = hm.cum_recharge / hm.cum_precip
df_output['(P-Q-Qgw)/P'] = (hm.cum_precip - hm.cum_runoff - hm.cum_gw_export)/hm.cum_precip
df_output['Q/P'] = hm.cum_runoff/hm.cum_precip

#load the full storage discharge dataset that was just generated
df = pd.read_csv('../post_proc/%s/dt_qs_s_%d_%d.csv'%(base_output_path, ID, IT), sep=',',header=None, names=['dt','r', 'qs', 'S', 'qs_cells'])
# remove the first row (test row written by init of gdp)
df.drop(0, inplace=True)
df.reset_index(drop=True, inplace=True)
# make dimensionless timeseries
Atot = np.sum(mg.cell_area_at_node[mg.core_nodes])
df['qs_star'] = df['qs']/(p*Atot)
df['S_star'] = df['S']/(b*ne*Atot)

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

# find times with recharge. Note in df qs and S are at the end of the timestep.
# i is at the beginning of the timestep. *** note now that recharge includes
# periods of negative recharge where extraction occurs.
df['t'] = np.cumsum(df['dt'])
is_r = df['r'] > 0.0
pre_r = np.where(np.diff(is_r*1)>0.0)[0] #last data point before recharge
end_r = np.where(np.diff(is_r*1)<0.0)[0][1:] #last data point where there is recharge

# make a linear-type baseflow separation, where baseflow during recharge increases
# linearly from its initial qs before recharge to its final qs after recharge.
qb = df['qs'].copy()
q = df['qs']
t = df['t']
for i in range(len(end_r)):
    j = pre_r[i]
    k = end_r[i]
    slope = (q[k+1] - q[j])/(t[k+1]-t[j])
    qb[j+1:k+1] = q[j]+slope*(t[j+1:k+1] - t[j])

df['qb'] = qb
qe = df['qs'] - df['qb']

# integrate to find total values. trapezoidal for the properties that
# change dynamically, and simple rectangular for i bc we know rain varies in this way.
df_output['qe_tot'] = np.sum(df['dt'] * qe)
df_output['qb_tot'] = np.sum(df['dt'] * qb)
df_output['qs_tot'] = np.sum(df['dt'] * df['qs'])
df_output['r_tot'] = np.sum(df['dt'] * df['r']) # recharge - extraction

# L'vovich partitioning: wetting on precipitation
df_output['W/P'] = (df_output['cum_precip'] - df_output['qe_tot'])/df_output['cum_precip']
df_output['Qb/W'] = df_output['qb_tot']/(df_output['cum_precip'] - df_output['qe_tot'])
df_output['Qb/Q'] = df_output['qb_tot']/df_output['qs_tot'] #baseflow index
# df_output['RR'] = qe_tot/r_tot #runoff ratio

###### spatial runoff related quantities

# effective Qstar
Q_all = hm.Q_all[1:,:]
dt = np.diff(hm.time)
intensity = hm.intensity[:-1]

# recharge
recharge_event = mg.add_zeros('node', 'recharge_rate_mean_storm')
recharge_event[:] = np.mean(hm.r_all[range(0,hm.r_all.shape[0],2),:], axis=0)

# extraction
extraction_interevent = mg.add_zeros('node', 'extraction_rate_mean_interstorm')
extraction_interevent[:] = np.mean(hm.e_all[range(2,hm.e_all.shape[0],2),:], axis=0)

# mean and variance of water table
wt_all = hm.wt_all[1:,:]
base_all = np.ones(wt_all.shape)*mg.at_node['aquifer_base__elevation']
elev_all = np.ones(wt_all.shape)*mg.at_node['topographic__elevation']
wtrel_all = np.zeros(wt_all.shape)
wtrel_all[:, mg.core_nodes] = (wt_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])/(elev_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])

# water table and saturation at end of storm and interstorm
sat_all = (elev_all-wt_all) < sat_cond*hg
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

# classify zones of saturation
sat_class = mg.add_zeros('node', 'saturation_class')
sat_never = np.logical_and(sat_end_storm < 0.05, sat_end_interstorm < 0.05)
sat_always = np.logical_and(sat_end_interstorm > 0.95, sat_end_storm > 0.95)
sat_variable = ~np.logical_or(sat_never, sat_always)
sat_class[sat_never] = 0
sat_class[sat_variable] = 1
sat_class[sat_always] = 2
df_output['sat_never'] = np.sum(sat_never[mg.core_nodes])/mg.number_of_core_nodes
df_output['sat_variable'] = np.sum(sat_variable[mg.core_nodes])/mg.number_of_core_nodes
df_output['sat_always'] = np.sum(sat_always[mg.core_nodes])/mg.number_of_core_nodes

#### saturtion probability and entropy
calc_entropy = lambda x: -x*np.log2(x) - (1-x)*np.log2(1-x)

# first method: time weighted variability in saturation
sat_prob = mg.add_zeros('node', 'saturation_probability')
sat_entropy = mg.add_zeros('node', 'saturation_entropy')
sat_prob[:] = np.sum((sat_all.T*dt)/np.sum(dt), axis=1)
sat_entropy[:] = calc_entropy(sat_prob)
sat_entropy[np.isnan(sat_entropy)] = 0.0
df_output['sat_entropy'] = np.sum(sat_entropy[mg.core_nodes])

# second method: interstorm-storm unsat-sat probability
sat_unsat_prob = mg.add_zeros('node', 'sat_unsat_union_probability')
# P(sat_storm and unsat_interstorm) = P(sat_storm given unsat_interstorm)*P(unsat_interstorm)
sat_storms = sat_all[intensity>0,:]; sat_interstorms = sat_all[intensity==0.0,:] # first storm record precedes interstorm record
sat_storms = sat_storms[1:,:]; sat_interstorms = sat_interstorms[:-1,:] # adjust indices so that prev interstorm is aligned with storm
p_unsat_interstorm = np.sum(~sat_interstorms, axis=0)/len(sat_interstorms) # prob unsaturated at the end of an interstorm
p_cond_sat_storm_unsat_interstorm = np.sum(sat_storms*~sat_interstorms, axis=0)/np.sum(~sat_interstorms, axis=0) # prob that saturated at end of storm given that it's unsaturated at end of interstorm
p_union_sat_storm_unsat_interstorm = p_cond_sat_storm_unsat_interstorm*p_unsat_interstorm # prob that unsaturated at end of interstorm and saturated at end of storm
sat_unsat_prob[:] = p_union_sat_storm_unsat_interstorm


##### channel network
#find number of saturated cells
count_sat_nodes = np.sum(sat_all,axis=1)
# find median channel network at end of storm and end of interstorm
network_id_sat_interstorm = np.where(count_sat_nodes == np.percentile(count_sat_nodes[intensity==0], 50, interpolation='nearest'))[0][0]

#set fields
network_curvature = mg.add_zeros('node', 'channel_mask_curvature')
network_sat = mg.add_zeros('node', 'channel_mask_sat_interstorm')
network_curvature[:] = curvature > 0
network_sat[:] = sat_all[network_id_sat_interstorm,:]


######## Calculate HAND
hand_curvature = mg.add_zeros('node', 'hand_curvature')
hand_sat = mg.add_zeros('node', 'hand_sat_interstorm')

hd = HeightAboveDrainageCalculator(mg, channel_mask=network_curvature)
try:
    hd.run_one_step()
    hand_curvature[:] = mg.at_node["height_above_drainage__elevation"].copy()
    df_output['mean_hand_curvature'] = np.mean(hand_curvature[mg.core_nodes])

    hd.channel_mask = network_sat
    hd.run_one_step()
    hand_sat[:] = mg.at_node["height_above_drainage__elevation"].copy()
    df_output['mean_hand_sat_interstorm'] = np.mean(hand_sat[mg.core_nodes])

except:
    print('failed to calculate HAND')

######## Calculate drainage density
dd = DrainageDensity(mg, channel__mask=np.uint8(network_curvature))
try:
    channel_mask = mg.at_node['channel__mask']
    df_output['dd_curvature'] = dd.calculate_drainage_density()
    df_output['mean hillslope len curvature'] = 1/(2*df_output['dd_curvature'])

    channel_mask[:] = np.uint8(network_sat)
    df_output['dd_sat_interstorm'] = dd.calculate_drainage_density()
    df_output['mean hillslope len sat interstorm'] = 1/(2*df_output['dd_sat_interstorm'])

except:
    print('failed to calculate drainage density')

####### save things

output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        'at_node:channel_mask_curvature',
        'at_node:channel_mask_sat_interstorm',
        'at_node:hand_curvature',
        'at_node:hand_sat_interstorm',
        'at_node:saturation_class',
        'at_node:saturation_probability',
        'at_node:saturation_entropy',
        'at_node:sat_unsat_union_probability',
        'at_node:topographic__index_D8',
        'at_node:topographic__index_D4',
        'at_node:slope_D8',
        'at_node:slope_D4',
        'at_node:drainage_area',
        'at_node:curvature',
        'at_node:steepness',
        'at_node:slope_rd',
        'at_node:profile_curvature_rd',
        'at_node:planform_curvature_rd',
        'at_node:total_curvature_rd',
        'at_node:surface_water_effective__discharge',
        'at_node:recharge_rate_mean_storm',
        'at_node:extraction_rate_mean_interstorm',
        'at_node:wtrel_mean_end_storm',
        'at_node:wtrel_mean_end_interstorm',
        'at_node:sat_mean_end_storm',
        'at_node:sat_mean_end_interstorm',
        'at_node:Q_mean_end_storm',
        'at_node:Q_mean_end_interstorm',
        ]

filename = '../post_proc/%s/grid_%d_%d.nc'%(base_output_path, ID, IT)
to_netcdf(mg, filename, include=output_fields, format="NETCDF4")

df_output = pd.DataFrame.from_dict(df_output, orient='index', columns=[ID])
df_output.to_csv('../post_proc/%s/output_ID_%d_%d.csv'%(base_output_path, ID, IT))
df.to_csv('../post_proc/%s/q_s_dt_ID_%d_%d.csv'%(base_output_path, ID, IT))
