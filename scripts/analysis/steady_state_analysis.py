"""
Analysis of results on HPC for steady stream power model runs.
"""

import os
import glob
from re import sub
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from landlab import imshow_grid
from landlab.io.netcdf import read_netcdf, write_raster_netcdf

from landlab import RasterModelGrid, LinkStatus
from landlab.components import (
    GroundwaterDupuitPercolator,
    HeightAboveDrainageCalculator,
    DrainageDensity,
    )
from DupuitLEM.auxiliary_models import HydrologySteadyStreamPower

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']

########## Load and basic plot
grid_files = glob.glob('./data/steady_state/*.nc')
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))

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

pickle.dump(df_z_change, open('../post_proc/%s/z_change_steady_state_%d.p'%(base_output_path, ID), 'wb'))
