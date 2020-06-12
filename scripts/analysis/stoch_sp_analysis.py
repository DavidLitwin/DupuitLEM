"""
Analysis of results on MARCC for stochastic stream power model runs.
"""

import os
import glob
from re import sub
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pandas import read_csv

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


base_output_path = 'stoch_sp_3'
ID = 11


############ basic plots
grid_files = glob.glob('C:/Users/dgbli/Documents/MARCC_output/DupuitLEMResults/stoch_sp_3-11/data/*.nc')
grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:float(sub("\D", "", x[50:-3])))

path = files[-1]
iteration = float(sub("\D", "", path[50:-3]))

grid = read_netcdf(path)
elev = grid.at_node['topographic__elevation']
base = grid.at_node['aquifer_base__elevation']
wt = grid.at_node['water_table__elevation']

# elevation
plt.figure(figsize=(8,6))
imshow_grid(grid,elev, cmap='gist_earth', colorbar_label = 'Elevation [m]', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID,iteration))
plt.savefig('../figs/'+base_output_path+'/elev_ID_%d.png'%ID)
plt.close()

# relative saturation
plt.figure(figsize=(8,6))
imshow_grid(grid,(wt-base)/(elev-base), cmap='Blues', limits=(0,1), colorbar_label = 'Relative saturated thickness [-]', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID,iteration))
plt.savefig('../figs/'+base_output_path+'/sat_ID_%d.png'%ID)
plt.close()

# water table
plt.figure(figsize=(8,6))
imshow_grid(grid,wt, cmap='Blues', colorbar_label = 'water table elevation', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID,iteration))
plt.savefig('../figs/'+base_output_path+'/wt_ID_%d.png'%ID)
plt.close()

# regolith thickness
# plt.figure(figsize=(8,6))
# imshow_grid(grid,elev-base, cmap='Blues', colorbar_label = 'regolith thickness', grid_units=('m','m'))
# plt.title('ID %d, Iteration %d'%(ID,iteration))
# plt.savefig('../figs/'+base_output_path+'/elev_ID_%d'%ID)
# plt.close()

################ find channel network
df_params = pickle.load(open('C:/Users/dgbli/Documents/MARCC_output/DupuitLEMResults/'+base_output_path+'-%d/parameters.p'%ID,'rb'))

# df_params = pickle.load(open('./parameters.p'%ID,'rb'))

Ks = df_params['ksat'][ID] #surface hydraulic conductivity [m/s]
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

f = open('dt_qs_s.csv', 'w')
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

pd = PrecipitationDistribution(mg, mean_storm_duration=storm_dt,
    mean_interstorm_duration=interstorm_dt, mean_storm_depth=p_d,
    total_t=T_h)
pd.seed_generator(seedval=2)

hm = HydrologyEventStreamPower(
        mg,
        precip_generator=pd,
        groundwater_model=gdp,
)

#run model
hm.run_step_record_state()

df = read_csv('dt_qs_s.csv', sep=',',header=None, names=['dt','qs','S'])


plt.figure()
plt.plot(np.cumsum(df['dt']/(24*3600)),df['S'])
plt.xlabel('time [d]')
plt.ylabel('storage $[m^3]$')

plt.figure()
plt.plot(np.cumsum(df['dt']/(24*3600)),df['qs'])
plt.xlabel('time [d]')
plt.ylabel('discharge $[m^3/s]$')

plt.figure()
plt.loglog(df['S'],df['qs'], '.')
plt.xlabel('storage $[m^3]$')
plt.ylabel('discharge $[m^3/s]$')


#find channel network
qs_all = hm.Q_all[30:,:]
sat_cells = qs_all > 1e-10
count_sat_cells = np.sum(sat_cells,axis=1)
min_network_id = np.where(count_sat_cells == min(count_sat_cells))[0][0]
max_network_id = np.where(count_sat_cells == max(count_sat_cells))[0][0]
med_network_id = np.where(count_sat_cells == np.median(count_sat_cells))[0][0]

min_network = sat_cells[min_network_id,:]
max_network = sat_cells[max_network_id,:]
med_network = sat_cells[med_network_id,:]

######## Runoff generation Analysis

time = hm.time[30:]
p = hm.intensity[30:]
qs_all = hm.qs_all[30:,:]
Q_all = hm.Q_all[30:,:]
qs_sum = 0 #total proportion of water leaving via surface [m]
p_sum = 0 #total precipitation [m]
qs_sum_event = 0 #qs during storm events only [m]
qe_sum = np.zeros(len(elev)) #cumulative exfiltration [m]
qp_sum = np.zeros(len(elev)) #cumulative precip on saturated area [m]
Q_sum = np.zeros(len(elev))
for i in range(np.shape(qs_all)[0]-1):

    qe = np.maximum(qs_all[i,:]-p[i],0)
    qe_sum += qe*(time[i+1]-time[i])

    qp = qs_all[i,:]-qe_all[i,:]
    qp_sum += qp*(time[i+1]-time[i])

    Q_sum += Q_all[i,:]*(time[i+1]-time[i])

    qs_sum += np.sum(qs_all[i,:]*(time[i+1]-time[i]))
    p_sum += p[i]*(time[i+1]-time[i])

    if p[i]>0.0:
        qs_sum_event += np.sum(qs_all[i,:]*(time[i+1]-time[i]))

        
BFI = (qs_sum - qs_sum_event)/qs_sum #this is *a* baseflow index, but not really *the* baseflow index.

# exfiltration proportion
plt.figure(figsize=(8,6))
imshow_grid(grid,qs_sum, cmap='Blues', colorbar_label = 'exfiltration [m]', grid_units=('m','m'))
plt.title('ID %d, Iteration %d'%(ID,iteration))
plt.savefig('../figs/'+base_output_path+'/exfilt_ID_%d.png'%ID)
plt.close()


plt.figure()
imshow_grid(grid,p_sat_sum, cmap = 'Blues', colorbar_label = 'precip on saturated area', grid_units=('m','m'))
plt.savefig('C:/Users/dgbli/Documents/MARCC_output/DupuitLEMResults/figs/stoch_vary_k_'+str(num)+'/p_sat_K=%.2f_d=%.2f.png'%(Ks*3600,d_k))


######## Calculate HAND

hd = HeightAboveDrainageCalculator(mg, channel_mask=min_network)

hd.run_one_step()
hand_min = mg.at_node["height_above_drainage__elevation"].copy()
mean_hand_min = np.mean(hand_min)

hd.channel_mask = max_network
hd.run_one_step()
hand_max = mg.at_node["height_above_drainage__elevation"].copy()
mean_hand_max = np.mean(hand_max)

hd.channel_mask = med_network
hd.run_one_step()
hand_med = mg.at_node["height_above_drainage__elevation"].copy()
mean_hand_med = np.mean(hand_med)

######## Caclculate drainage density

dd = DrainageDensity(mg, channel__mask=min_network)
channel_mask = mg.at_node['channel__mask']
dd_min = dd.calculate_drainage_density()

channel_mask[:] = max_network
dd_max = dd.calculate_drainage_density()

channel_mask[:] = med_network
dd_med = dd.calculate_drainage_density()

####### save things

ch_min = mg.add_zeros('node', 'channel_mask_min')
ch_max = mg.add_zeros('node', 'channel_mask_max')
ch_med = mg.add_zeros('node', 'channel_mask_med')
ch_min[:] = min_network
ch_max[:] = max_network
ch_med[:] = med_network

hd_min = mg.add_zeros('node', 'hand_min')
hd_max = mg.add_zeros('node', 'hand_max')
hd_med = mg.add_zeros('node', 'hand_med')
hd_min[:] = hand_min
hd_max[:] = hand_max
hd_med[:] = hand_med

p = mg.add_zeros('node', 'cum_precipitation')
qs = mg.add_zeros('node', 'cum_surface_water__specific_discharge')
qe = mg.add_zeros('node', 'cum_exfiltration')
qp = mg.add_zeros('node', 'cum_precip_sat')
Q = mg.add_zeros('node', 'cum_discharge')
runon = mg.add_zeros('node', 'cum_runon')
pcp[:] = p_sum
qs[:] = qs_sum
qe[:] = qe_sum
qp[:] = qp_sum
Q[:] = Q_sum
runon[:] = runon_sum


output_fields = [
        "topographic__elevation",
        "aquifer_base__elevation",
        "water_table__elevation",
        'channel_mask_min',
        'channel_mask_max',
        'channel_mask_med',
        'hand_min',
        'hand_max',
        'hand_med',
        ]

filename = '../../../DupuitLEMResults/figs/'+base_output_path+'/grid_ID_%d.nc'%ID
write_raster_netcdf(filename, mg, names = output_fields, format="NETCDF4")
