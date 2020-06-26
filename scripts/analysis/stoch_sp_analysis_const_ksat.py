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

from landlab import RasterModelGrid
from landlab.components import (
    GroundwaterDupuitPercolator,
    PrecipitationDistribution,
    HeightAboveDrainageCalculator,
    DrainageDensity,
    )
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

Ks = 1e-5 #surface hydraulic conductivity [m/s]
K0 = Ks*0.01 #minimum hydraulic conductivity [m/s]
n = df_params['n'][ID] #drainable porosity [-]
beq = df_params['beq'][ID] #characteristic depth  [m]
storm_dt = df_params['storm_dt'][ID] #mean storm duration [s]
interstorm_dt = df_params['interstorm_dt'][ID] #mean interstorm duration [s]
p_d = df_params['depth'][ID] #mean storm depth [m]
T_h = 365*24*3600 #total hydrological time [s]

#initialize grid
dx = 10.0
mg = RasterModelGrid((100, 100), xy_spacing=dx)
mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_CLOSED, top=mg.BC_NODE_IS_CLOSED, \
                              left=mg.BC_NODE_IS_FIXED_VALUE, bottom=mg.BC_NODE_IS_CLOSED)
z = mg.add_zeros('node', 'topographic__elevation')
z[:] = elev
zb = mg.add_zeros('node', 'aquifer_base__elevation')
zb[:] = base
zwt = mg.add_zeros('node', 'water_table__elevation')
zwt[:] = wt

#initialize landlab components
ksat_fun = bind_avg_hydraulic_conductivity(Ks,K0,beq) # hydraulic conductivity [m/s]

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
                                  courant_coefficient=0.1,
                                  vn_coefficient = 0.1,
                                  callback_fun = write_SQ,
                                  )

pdr = PrecipitationDistribution(mg, mean_storm_duration=storm_dt,
    mean_interstorm_duration=interstorm_dt, mean_storm_depth=p_d,
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

# S_test = np.linspace(min(S),max(S),100)
# plt.figure()
# plt.plot(S,Q, '.')
# plt.plot(S_test,power_law(S_test, *pars), '--')
# plt.xlabel('storage $[m^3]$')
# plt.ylabel('discharge $[m^3/s]$')

# plt.figure()
# plt.plot(np.cumsum(df['dt']/(24*3600)),df['S'])
# plt.xlabel('time [d]')
# plt.ylabel('storage $[m^3]$')

# plt.figure()
# plt.plot(np.cumsum(df['dt']/(24*3600)),df['qs'])
# plt.xlabel('time [d]')
# plt.ylabel('discharge $[m^3/s]$')


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


qe_sum = mg.add_zeros('node', 'cum_exfiltration') #cumulative exfiltration [m]
qp_sum = mg.add_zeros('node', 'cum_precip_sat') #cumulative precip on saturated area [m]
for i in range(np.shape(qs_all)[0]):

    qe = np.maximum(qs_all[i,:]-p[i],0)
    qe_sum += qe*dt_q[i]

    qp = qs_all[i,:]-qe
    qp_sum += qp*dt_q[i]

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
        'cum_precip_sat'
        ]

filename = '../post_proc/%s/grid_%d.nc'%(base_output_path, ID)
write_raster_netcdf(filename, mg, names = output_fields, format="NETCDF4")

pickle.dump(df_output, open('../post_proc/%s/output_ID_%d.p'%(base_output_path, ID), 'wb'))
