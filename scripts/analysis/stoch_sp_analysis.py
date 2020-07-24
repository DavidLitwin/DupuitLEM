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

from landlab import imshow_grid
from landlab.io.netcdf import read_netcdf, write_raster_netcdf

from landlab import RasterModelGrid, LinkStatus
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    HeightAboveDrainageCalculator,
    DrainageDensity,
    )
from landlab.grid.mappers import map_max_of_node_links_to_node
from DupuitLEM.auxiliary_models import HydrologyEventStreamPower
from DupuitLEM.grid_functions.grid_funcs import bind_avg_hydraulic_conductivity

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

Ks = df_params['ksat'][ID] #surface hydraulic conductivity [m/s]
K0 = Ks*0.01 #minimum hydraulic conductivity [m/s]
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

#initialize landlab components
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,b) # hydraulic conductivity [m/s]

f = open('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), 'w')
def write_SQ(grid,dt,file=f):
    cores = grid.core_nodes
    h = grid.at_node["aquifer__thickness"]
    area = grid.cell_area_at_node
    storage = np.sum(n*h[cores]*area[cores])

    qs = grid.at_node["surface_water__specific_discharge"]
    qs_tot = np.sum(qs*area)

    file.write('%f, %f, %f\n'%(dt,qs_tot,storage))

gdp = GroundwaterDupuitPercolator(mg,
                                  porosity=n,
                                  hydraulic_conductivity=ksat_fun,
                                  regularization_f=0.01,
                                  recharge_rate=0.0,
                                  courant_coefficient=0.01*Ks/1e-5,
                                  vn_coefficient = 0.01*Ks/1e-5,
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
df = pd.read_csv('../post_proc/%s/dt_qs_s_%d.csv'%(base_output_path, ID), sep=',',header=None, names=['dt','qs','S'])

##### recession
def power_law(x, a, b, c):
    return a*np.power(x, b) + c

rec_inds = np.where(np.diff(df['qs'], prepend=0.0) < 0.0)[0]
Q = df['qs'][rec_inds]
S = df['S'][rec_inds] - min(df['S'][rec_inds])

pars, cov = curve_fit(f=power_law, xdata=S, ydata=Q, p0=[0, 1, 0], bounds=(-100, 100))
stdevs = np.sqrt(np.diag(cov))

def linear_law(x,a,c):
    return a*x + c

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

##### channel network
qs_all = hm.Q_all[30:,:]
sat_cells = qs_all > 1e-10
count_sat_cells = np.sum(sat_cells,axis=1)
min_network_id = np.where(count_sat_cells == min(count_sat_cells))[0][0]
max_network_id = np.where(count_sat_cells == max(count_sat_cells))[0][0]
med_network_id = np.where(count_sat_cells == np.median(count_sat_cells))[0][0]

min_network = mg.add_zeros('node', 'channel_mask_min')
max_network = mg.add_zeros('node', 'channel_mask_max')
med_network = mg.add_zeros('node', 'channel_mask_med')

min_network[:] = sat_cells[min_network_id,:]
max_network[:] = sat_cells[max_network_id,:]
med_network[:] = sat_cells[med_network_id,:]

##### steepness and curvature
S = mg.add_zeros('node', 'slope')
A = mg.at_node['drainage_area']
curvature = mg.add_zeros('node', 'curvature')
steepness = mg.add_zeros('node', 'steepness')

dzdx = grid.calc_grad_at_link(z)
dzdx[mg.status_at_link == LinkStatus.INACTIVE] = 0.0
S[:] = map_max_of_node_links_to_node(mg,abs(dzdx))
curvature[:] = mg.calc_flux_div_at_node(dzdx)
steepness[:] = np.sqrt(A)*S

######## Runoff generation
areas = mg.cell_area_at_node #areas on the grid
# timeseries recorded, assuming more or less equilibrated after 30 timesteps
time = hm.time[30:]
p = hm.intensity[30:]
qs_all = hm.qs_all[30:,:]

#offset because p is recorded at beginning of dt and q at the end
dt_q = np.diff(time,prepend=time[0])
dt_p = np.diff(time,append=time[-1])
qs = np.sum(qs_all*areas,axis=1)

#total fluxes
p_sum = np.sum(p*dt_p)*np.sum(areas) # total water entering as precipitation [m3]
qs_sum = np.sum(qs*dt_q) # total water leaving via surface [m3]
qs_sum_event = np.sum(qs[p>0]*dt_q[p>0])

df_output['BFI'] = (qs_sum - qs_sum_event)/qs_sum # this is *a* baseflow index, but not really *the* baseflow index.
df_output['RR'] = qs_sum/p_sum # runoff ratio

#separate exfiltration and precip on sat area
qe_sum = mg.add_zeros('node', 'cum_exfiltration') #cumulative exfiltration [m]
qp_sum = mg.add_zeros('node', 'cum_precip_sat') #cumulative precip on saturated area [m]
for i in range(np.shape(qs_all)[0]):

    qe = np.maximum(qs_all[i,:]-p[i],0)
    qe_sum += qe*dt_q[i]

    qp = qs_all[i,:]-qe
    qp_sum += qp*dt_q[i]

#quantiles of Q*
r = np.arange(hm.Q_all.shape[0])
Q_all = hm.Q_all[r%2==1,:]
Q_all_max = np.max(Q_all,axis=1)
percs = [100,90,50,10,0]
Q_star_percs = np.zeros((Q_all.shape[1],len(percs)))
for i in range(len(percs)):
    index = np.where(Q_all_max==np.percentile(Q_all_max,percs[i], interpolation='nearest'))[0][0]
    Q_star_percs[:,i] = Q_all[index,:]/(mg.at_node['drainage_area']*df_params['p'][ID])
df_qstar = pd.DataFrame(data=Q_star_percs, columns=percs)
df_qstar['mean'] = np.mean(Q_all,axis=0)/(mg.at_node['drainage_area']*df_params['p'][ID])
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
# TI_nd = TI/(df_params['l']/df_params['gam']) #nondimensionalized TI

crit_twi = mg.add_zeros('node', 'TI_exceedence_contour')
twi_contour = df_params['ksat'][ID]/(df_output['rec_a_linear']*df_params['n'][ID])
crit_twi[:] = TI >= twi_contour


####### calculate elevation change
z_change = np.zeros((len(files),4))
grid = read_netcdf(files[0])
elev0 = grid.at_node['topographic__elevation']
for i in range(1,len(files)):

    grid = read_netcdf(files[i])
    elev = grid.at_node['topographic__elevation']

    elev_diff = abs(elev-elev0)
    z_change[i,0] = np.max(elev_diff)
    z_change[i,1] = np.percentile(elev_diff,90)
    z_change[i,2] = np.percentile(elev_diff,50)
    z_change[i,3] = np.mean(elev_diff)

    elev0 = elev.copy()

df_z_change = pd.DataFrame(z_change,columns=['max', '90 perc', '50 perc', 'mean'])
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
        'cum_exfiltration',
        'cum_precip_sat',
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
