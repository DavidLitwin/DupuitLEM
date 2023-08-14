# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:34:32 2020

@author: dgbli
"""

import numpy as np
import pickle
import glob

from landlab import LinkStatus
from landlab.grid.mappers import map_downwind_node_link_max_to_node
from landlab.io.netcdf import from_netcdf, to_netcdf

from landlab.components import (
    HeightAboveDrainageCalculator,
    DrainageDensity,
    FlowAccumulator,
    )

base_output_path = 'simple_lem_4_5'
directory = 'C:/Users/dgbli/Documents/Research Data/HPC output/DupuitLEMResults/'
id_range = range(6)


#%%
for i in id_range:
    
    df_output = {}
    # mg = from_netcdf('%s/%s/grid_%d.nc'%(directory, base_output_path, i))
    
    grid_files = glob.glob('%s/%s-%d/data/*.nc'%(directory,base_output_path,i))
    files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
    iteration = int(files[-1].split('_')[-1][:-3])
    mg = from_netcdf(files[-1])
    print(files[-1])
    
    elev = mg.at_node['topographic__elevation']
    mg.set_status_at_node_on_edges(right=mg.BC_NODE_IS_CLOSED,
                                top=mg.BC_NODE_IS_CLOSED,
                                left=mg.BC_NODE_IS_FIXED_VALUE,
                                bottom=mg.BC_NODE_IS_CLOSED,
    )
    fa = FlowAccumulator(mg,flow_director='D8', depression_finder='DepressionFinderAndRouter')
    fa.run_one_step()
    
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
    
    network = mg.add_zeros('node', 'channel_mask')
    network[:] = curvature > 0
    
    ######## Calculate HAND
    hand = mg.add_zeros('node', 'hand')
    hd = HeightAboveDrainageCalculator(mg, channel_mask=network)
    
    hd.run_one_step()
    hand[:] = mg.at_node["height_above_drainage__elevation"].copy()
    df_output['mean_hand'] = np.mean(hand[mg.core_nodes])
    df_output['hand_mean_ridges'] = np.mean(hand[mg.at_node["drainage_area"]==mg.dx**2])
    
    ######## Calculate drainage density
    dd = DrainageDensity(mg, channel__mask=np.uint8(network))
    channel_mask = mg.at_node['channel__mask']
    df_output['drainage_density'] = dd.calculate_drainage_density()
    
    
    output_fields = [
        "at_node:topographic__elevation",
        'at_node:topographic__index_D8',
        'at_node:topographic__index_D4',
        'at_node:channel_mask',
        'at_node:hand',
        'at_node:slope_D8',
        'at_node:slope_D4',
        'at_node:drainage_area',
        'at_node:curvature',
        'at_node:steepness',
        ]

    filename = '%s/post_proc/%s/grid_%d.nc'%(directory, base_output_path,i)
    to_netcdf(mg, filename, include=output_fields, format="NETCDF4")
    
    pickle.dump(df_output, open('%s/post_proc/%s/output_%d.p'%(directory, base_output_path,i), 'wb'))
