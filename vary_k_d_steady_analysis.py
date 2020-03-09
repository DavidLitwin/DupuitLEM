# -*- coding: utf-8 -*-
"""
Created on Mar 9, 2020

Analysis of the output of steady recharge version of vary_k_d. Make static plots,
calculate drainage density and recession constants.

@author: dgbli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import pickle
from itertools import product

from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

from landlab import imshow_grid
from landlab.io.netcdf import read_netcdf
from landlab.components import DrainageDensity, FlowAccumulator, GroundwaterDupuitPercolator
from landlab.grid.mappers import map_mean_of_link_nodes_to_link

def avg_hydraulic_conductivity(grid,h,b,k0,ks,dk):
    """
    Calculate the average hydraulic conductivity when hydraulic conductivity
    varies with depth as:

        k = k0 + (ks-k0)*exp(-d/dk)

    Parameters:
        h = aquifer thickness
        b = depth from surface to impermeable base
        k0: asymptotic permeability at infinite depth
        ks: hydraulic conductivity at the ground surface
        dk = characteristic depth

    """

    blink = map_mean_of_link_nodes_to_link(grid,b)
    hlink = map_mean_of_link_nodes_to_link(grid,h)
    b1 = blink[hlink>0.0]
    h1 = hlink[hlink>0.0]
    kavg = np.zeros_like(hlink)
    kavg[hlink>0.0] = dk*(ks-k0)/h1 * (np.exp(-(b1-h1)/dk) - np.exp(-b1/dk)) + k0
    return kavg

#from twi_analysis_funcs
def log_Recession_constant(X_0):
    Q_0 = []
    Q_1 = []
    for i in range(1,X_0.shape[0]):
        if X_0[i] < X_0[i-1]:
            Q_0.append(X_0[i-1])
            Q_1.append(X_0[i])
    Q_0 = np.array(Q_0)
    Q_1 = np.array(Q_1)

    Q_0_bool = Q_0 != 0
    Q_1_bool = Q_1 != 0
    Q_bool = np.logical_and(Q_0_bool, Q_1_bool)

    Q_1_new = np.log10(np.extract(Q_bool,Q_1))
    Q_0_new = np.log10(np.extract(Q_bool,Q_0))

    A = np.vstack([Q_0_new, np.ones(len(Q_0_new))]).T
    a,c = np.linalg.lstsq(A,Q_1_new,rcond=None)[0]
    return a, c, Q_0_new, Q_1_new

###################### initialize
# Set parameters
R = 1.5/(365*24*3600)  # steady, uniform recharge rate [m/s]
Ks_all = np.array([0.2, 1, 2])/(3600)  # hydraulic conductivity at the surface [m/s]
w0 = 2E-4/(365*24*3600) #max rate of soil production [m/s]
d_i_rel = 1.0 # initial depth relative to steady state depth [-]
d_s_all = [0.5,1.0,2.0] # characteristic soil production depth [m]
uplift_rate = 1E-4/(365*24*3600) # uniform uplift [m/s]
m = 0.5 #Exponent on Q []
n = 1.0 #Exponent on S []
K = 5E-8 #erosivity coefficient [m-1/2 s−1/2]
D = 0.01/(365*24*3600) # hillslope diffusivity [m2/s]

params = np.array(list(product(Ks_all,d_s_all)))

#######################

mean_drainage_densities = np.zeros((len(params),3))
recession_k = np.zeros((len(params),2))
IDs = np.zeros(len(params))
Ks_save = np.zeros(len(params))
paths = glob.glob('../DupuitLEMResults/vary_k_d/vary_k_d_9vals_steady*')

for i in range(len(paths)):

    # find the last file
    files = glob.glob(paths[i]+'/data/*.nc')
    ID = int(re.sub("[^0-9]", "", paths[i][-2:]))
    IDs[i] = ID
    Ks_save[i] = params[ID,0]*3600
    Ks = params[ID,0]
    K0 = 0.01*Ks
    d_s = params[ID,1]
    d_k = d_s
    max_num = 0
    for j in range(len(files)):
        num = int(re.sub("[^0-9]", "", files[j][-9:]))

        if num > max_num:
            max_num = num
            max_file = files[j]


    # import
    grid = read_netcdf(max_file)
    elev = grid.at_node['topographic__elevation']
    base = grid.at_node['aquifer_base__elevation']
    wt = grid.at_node['water_table__elevation']

    # surface elevation
    plt.figure(figsize=(8,6))
    imshow_grid(grid,'topographic__elevation', cmap='gist_earth', colorbar_label = 'Elevation [m]', grid_units=('m','m'))
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/elev_'+str(ID) +'.png')
    plt.close()

    # regolith thickness
    plt.figure(figsize=(8,6))
    imshow_grid(grid,grid.at_node['topographic__elevation'] - grid.at_node['aquifer_base__elevation'],cmap='YlOrBr', colorbar_label = 'Regolith thickness [m]', grid_units=('m','m'))
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/soil_'+str(ID) +'.png')
    plt.close()

    # relative saturation
    plt.figure(figsize=(8,6))
    imshow_grid(grid,(wt-base)/(elev-base), cmap='Blues', limits=(0,1), colorbar_label = 'Relative saturated thickness [-]', grid_units=('m','m'))
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/rel_thickness_'+str(ID) +'.png')
    plt.close()

    # surface water discharge
    plt.figure(figsize=(8,6))
    imshow_grid(grid,grid.at_node['surface_water__discharge'], cmap='plasma', colorbar_label = 'surface water discharge [m3/s]', grid_units=('m','m'))
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/surface_water_'+str(ID) +'.png')
    plt.close()

    # cross sections
    y_node = [100,500,900]
    fig,axs = plt.subplots(3,1,figsize=(8,6))
    for k in range(3):
        middle_row = np.where(grid.x_of_node == y_node[2-k])[0][1:-1]

        axs[k].fill_between(grid.y_of_node[middle_row],elev[middle_row],base[middle_row],facecolor=(198/256,155/256,126/256) )
        axs[k].fill_between(grid.y_of_node[middle_row],wt[middle_row],base[middle_row],facecolor=(145/256,176/256,227/256), alpha=0.8 )
        axs[k].fill_between(grid.y_of_node[middle_row],base[middle_row],np.zeros_like(base[middle_row]),facecolor=(111/256,111/256,111/256) )

    axs[2].set_xlabel('Distance (m)')
    axs[2].set_ylabel('Elevation (m)')
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/cross_section_'+str(ID) +'.png')
    plt.close()


    # relative change plot
    file1 = glob.glob(paths[i]+'/data/*90perc_rel_change.txt')
    file2 = glob.glob(paths[i]+'/data/*max_rel_change.txt')
    rel_change_90 = np.loadtxt(file1[0])
    rel_change_max = np.loadtxt(file2[0])

    ind = rel_change_90>0
    rel_change_90 = rel_change_90[ind]
    rel_change_max = rel_change_max[ind]

    plt.figure(figsize=(8,6))
    plt.semilogy(rel_change_max,label='Maximum')
    plt.semilogy(rel_change_90,label='90th percentile')
    plt.ylabel('Relative elevation change')
    plt.xlabel('Time step')
    plt.legend()
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/log_rel_change_'+str(ID)+'.png')
    plt.close()

    ################################# Recession and drainage density

    grid = read_netcdf(max_file)
    grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED, top=grid.BC_NODE_IS_CLOSED, \
                              left=grid.BC_NODE_IS_FIXED_VALUE, bottom=grid.BC_NODE_IS_CLOSED)
    elev = grid.at_node['topographic__elevation']
    base = grid.at_node['aquifer_base__elevation']
    wt = grid.at_node['water_table__elevation']
    Q_steady = grid.at_node['surface_water__discharge'].copy()

    # initialize model components
    gdp = GroundwaterDupuitPercolator(grid, porosity=0.2, \
                                      recharge_rate=0.0,regularization_f=0.01)
    fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8',  \
                         depression_finder = 'DepressionFinderAndRouter', runoff_rate='average_surface_water__specific_discharge')

    # Run recession
    T = 30*24*3600
    dt = 2*60*60
    N = T//dt

    q_sw_out = np.zeros(N)
    network_size = np.zeros(N)
    for n in range(N):
        gdp.K = avg_hydraulic_conductivity(grid,grid.at_node['aquifer__thickness'],
                                         grid.at_node['topographic__elevation']-
                                         grid.at_node['aquifer_base__elevation'],
                                         K0,Ks,d_k,
                                         )
        #run gw model
        gdp.recharge = 0
        gdp.run_with_adaptive_time_step_solver(dt)
        fa.run_one_step()

        network_size[n] = sum(grid.at_node['surface_water__specific_discharge']>0)/grid.number_of_core_nodes
        q_sw_out[n] = gdp.calc_sw_flux_out()

        # active drainage network after 1 day
        if n == 12:
            Q_recession_1 = grid.at_node['surface_water__discharge'].copy()

    # active drainage network after 30 days
    Q_recession_30 = grid.at_node['surface_water__discharge'].copy()

    t = np.array(np.arange(0,T,dt))/3600
    plt.figure()
    plt.plot(t,q_sw_out)
    plt.xlabel('time [hr]')
    plt.ylabel('Surface water discharge [$m^3/s$]')
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/recession_time_'+str(ID) +'.png')
    plt.close()

    plt.figure()
    plt.plot(t, network_size*100)
    plt.xlabel('time [hr]')
    plt.ylabel('% nodes contributing surface water discharge')
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/recession_channels_'+str(ID) +'.png')
    plt.close()

    [a, c, Q_0_new, Q_1_new] = log_Recession_constant(q_sw_out) #a, c, Q_0_new, Q_1_new
    recession_k[i,:] = np.array([a,c])

    x = np.linspace(min(Q_0_new),max(Q_0_new),50)
    y = a*x+c
    plt.figure(figsize=(6,6))
    plt.loglog(10**(Q_0_new),10**(Q_1_new),'.',color='blue', alpha = 0.2)
    plt.loglog(10**(x),10**(y),'-',color = 'red')
    plt.xlabel('$Q_0 \, mm/hr$')
    plt.ylabel('$Q_1 \, mm/hr$')
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/recession_plot_'+str(ID) +'.png', bbox_inches = 'tight')
    plt.close()

    # calculate drainage densities
    steady_channels = np.array(Q_steady>= R*grid.dx*grid.dy, dtype=np.uint8)
    recession_channels_1 = np.array(Q_recession_1>= R*grid.dx*grid.dy, dtype=np.uint8)
    recession_channels_30 = np.array(Q_recession_30>= R*grid.dx*grid.dy, dtype=np.uint8)

    steady_dd = DrainageDensity(grid,channel__mask=steady_channels)
    steady_dd_mean = steady_dd.calc_drainage_density()

    recession_dd_1 = DrainageDensity(grid,channel__mask=recession_channels_1)
    recession_dd_mean_1 = recession_dd_1.calc_drainage_density()

    recession_dd_30 = DrainageDensity(grid,channel__mask=recession_channels_30)
    recession_dd_mean_30 = recession_dd_30.calc_drainage_density()

    mean_drainage_densities[i,0] = steady_dd_mean
    mean_drainage_densities[i,1] = recession_dd_mean_1
    mean_drainage_densities[i,2] = recession_dd_mean_30

    plt.figure()
    imshow_grid(grid,steady_channels, plot_name='Steady channel extent', allow_colorbar=False, cmap='Blues', grid_units=('m','m'))
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/steady_channels_'+str(ID) +'.png')
    plt.close()

    plt.figure()
    imshow_grid(grid,recession_channels_30, plot_name='30 day recession channel extent', allow_colorbar=False, cmap='Blues', grid_units=('m','m'))
    plt.savefig('../DupuitLEMResults/figs/vary_k_d_steady/min_channels_'+str(ID) +'.png')
    plt.close()

data = {'ID':IDs, 'Ks':Ks_save, 'ds':params[:,2], 'DD_steady':mean_drainage_densities[:,0], 'DD_1':mean_drainage_densities[:,1], 'DD_30':mean_drainage_densities[:,2], 'rec_a':recession_k[:,0], 'rec_c':recession_k[:,1] }
df = pd.DataFrame(data)
pickle.dump(df,open('../DupuitLEMResults/figs/vary_k_d_steady/data_processed.p','wb'))
