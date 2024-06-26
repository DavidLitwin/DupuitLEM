"""
Analysis of results on HPC for stochastic stream power model runs.

update for new stochastic models
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
    ChiFinder,
    )
from landlab.grid.mappers import map_downwind_node_link_max_to_node
from DupuitLEM.auxiliary_models import HydrologyEventVadoseStreamPower, SchenkVadoseModel
from DupuitLEM.grid_functions import bind_avg_exp_ksat, bind_avg_recip_ksat

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
try:
    df_params = pd.read_csv('parameters.csv', index_col=0)[task_id]
    df_params.to_csv('../post_proc/%s/params_ID_%d.csv'%(base_output_path,ID), index=True)
except FileNotFoundError:
    print('Could not find parameters.csv. Trying to find a 1D params file.')
    df_params_1d = pd.read_csv('df_params_1d_%d.csv'%ID, index_col=0)[task_id]

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
T_h = 2000*(tr+tb) #20*df_params['Th'] #total hydrological time [s]
sat_cond = 0.025 # distance from surface (units of hg) for saturation
conc = 0.5 # reference concacvity for chi analysis

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
zwt[:] = wt

#initialize components
if isinstance(mg, RasterModelGrid):
    method = 'D8'
elif isinstance(mg, HexModelGrid):
    method = 'Steepest'
else:
    raise TypeError("grid should be Raster or Hex")

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

#run model
hm.run_step()

f = open('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), 'w')
def write_SQ(grid, r, dt, file=f):
    cores = grid.core_nodes
    h = grid.at_node["aquifer__thickness"]
    wt = grid.at_node["water_table__elevation"]
    z = grid.at_node["topographic__elevation"]
    sat = (z-wt) < sat_cond*hg
    qs = grid.at_node["surface_water__specific_discharge"]
    area = grid.cell_area_at_node

    storage = np.sum(ne*h[cores]*area[cores])
    qs_tot = np.sum(qs[cores]*area[cores])
    sat_nodes = np.sum(sat[cores])
    r_tot = np.sum(r[cores]*area[cores])

    file.write('%f, %f, %f, %f, %f\n'%(dt, r_tot, qs_tot, storage, sat_nodes))
gdp.callback_fun = write_SQ

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
# z = mg.at_node['topographic__elevation']
# z[mg.boundary_nodes] = np.nan
# zrd = rd.rdarray(z.reshape(mg.shape), no_data=-9999)
# zrd.geotransform = [0.0, mg.dx, 0.0, 0.0, 0.0, mg.dx]

# profile_curvature = rd.TerrainAttribute(zrd, attrib='profile_curvature')
# planform_curvature = rd.TerrainAttribute(zrd, attrib='planform_curvature')
# tot_curvature = rd.TerrainAttribute(zrd, attrib='curvature')
# slope = rd.TerrainAttribute(zrd, attrib='slope_riserun')

# slp = mg.add_zeros('node', "slope_rd")
# slp[:] = slope.reshape(z.shape)

# pro = mg.add_zeros('node', "profile_curvature_rd")
# pro[:] = -profile_curvature.reshape(z.shape) #flip sign of profile curv

# plan = mg.add_zeros('node', "planform_curvature_rd")
# plan[:] = planform_curvature.reshape(z.shape)

# curv = mg.add_zeros('node', "total_curvature_rd")
# curv[:] = tot_curvature.reshape(z.shape)

######## Runoff generation
df_output['cum_precip'] = hm.cum_precip
df_output['cum_recharge'] = hm.cum_recharge
df_output['cum_runoff'] = hm.cum_runoff
df_output['cum_gw_export'] = hm.cum_gw_export
df_output['cum_pet'] = hm.cum_pet

"""ratio of total recharge to total precipitation, averaged over space and time.
this accounts for time varying recharge with precipitation rate, unsat
storage and ET, as well as spatially variable recharge with water table depth.
"""
df_output['recharge_efficiency'] = hm.cum_recharge / hm.cum_precip
df_output['(P-Q-Qgw)/P'] = (hm.cum_precip - hm.cum_runoff - hm.cum_gw_export)/hm.cum_precip
df_output['Q/P'] = hm.cum_runoff/hm.cum_precip

#load the full storage discharge dataset that was just generated
df = pd.read_csv('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), sep=',',header=None, names=['dt','r', 'qs', 'S', 'sat_nodes'])
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
# i is at the beginning of the timestep.
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
df_output['r_tot'] = np.sum(df['dt'] * df['r']) # recharge

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

# mean and variance of water table
wt_all = hm.wt_all[1:,:]
base_all = np.ones(wt_all.shape)*mg.at_node['aquifer_base__elevation']
elev_all = np.ones(wt_all.shape)*mg.at_node['topographic__elevation']
wtrel_all = np.zeros(wt_all.shape)
wtrel_all[:, mg.core_nodes] = (wt_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])/(elev_all[:, mg.core_nodes] - base_all[:, mg.core_nodes])

wtrel_05 = mg.add_zeros('node', 'wtrel_05')
wtrel_95 = mg.add_zeros('node', 'wtrel_95')
Srel = np.mean(wtrel_all, axis=1) #relative storage timeseries
quantiles = np.quantile(Srel, [0.05, 0.95])
quan = [(np.abs(Srel - i)).argmin() for i in quantiles]
wtrel_05[:] = wtrel_all[quan[0], :]
wtrel_95[:] = wtrel_all[quan[1], :]

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

######## Calculate drainage density and chi
dd = DrainageDensity(mg, channel__mask=np.uint8(network_curvature))
try:
    channel_mask = mg.at_node['channel__mask']
    df_output['dd_curvature'] = dd.calculate_drainage_density()
    df_output['mean hillslope len curvature'] = 1/(2*df_output['dd_curvature'])

    channel_mask[:] = np.uint8(network_sat)
    df_output['dd_sat_interstorm'] = dd.calculate_drainage_density()
    df_output['mean hillslope len sat interstorm'] = 1/(2*df_output['dd_sat_interstorm'])

    # chi
    cf = ChiFinder(mg, min_drainage_area=mg.dx**2, reference_concavity=conc, reference_area=1)
    cf.calculate_chi()

except:
    print('failed to calculate drainage density and chi')

####### calculate elevation change
try:
    output_interval = df_params['output_interval']
except KeyError:
    print('output_interval not in params table. Using default.')
    output_interval = (10/(dtg/tg)).round().astype(int)

dt = output_interval*dtg

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
r_change['r_nd'] = relief_change[:,0]/hg
r_change['drdt_nd'] = relief_change[:,1]*(tg/hg)
r_change['t_nd'] = np.arange(len(files))*(dt/tg)


####### save things

output_fields = [
        "at_node:topographic__elevation",
        "at_node:aquifer_base__elevation",
        'at_node:channel_mask_curvature',
        'at_node:channel_mask_sat_interstorm',
        'at_node:channel__chi_index',
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
        # 'at_node:slope_rd',
        # 'at_node:profile_curvature_rd',
        # 'at_node:planform_curvature_rd',
        # 'at_node:total_curvature_rd',
        'at_node:surface_water_effective__discharge',
        'at_node:recharge_rate_mean_storm',
        'at_node:wtrel_mean_end_storm',
        'at_node:wtrel_mean_end_interstorm',
        'at_node:wtrel_05',
        'at_node:wtrel_95',
        'at_node:sat_mean_end_storm',
        'at_node:sat_mean_end_interstorm',
        'at_node:Q_mean_end_storm',
        'at_node:Q_mean_end_interstorm',
        'at_node:flow__receiver_node',
        ]

filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
to_netcdf(mg, filename, include=output_fields, format="NETCDF4")

df_output = pd.DataFrame.from_dict(df_output, orient='index', columns=[ID])
df_output.to_csv('../post_proc/%s/output_ID_%d.csv'%(base_output_path, ID))
df.to_csv('../post_proc/%s/q_s_dt_ID_%d.csv'%(base_output_path, ID))
df_z_change.to_csv('../post_proc/%s/z_change_%d.csv'%(base_output_path, ID))
r_change.to_csv('../post_proc/%s/relief_change_%d.csv'%(base_output_path, ID))
