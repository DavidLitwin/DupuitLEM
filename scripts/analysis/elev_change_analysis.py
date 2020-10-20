"""
Analysis of elevation change

Was written for steady state scheme, now changed to just look at regular runs.
"""

import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from landlab.io.netcdf import read_netcdf, from_netcdf
from matplotlib.colors import LightSource
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']

########## Load
# grid_files = glob.glob('./data/steady_state/*.nc')
grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
df_params = pickle.load(open('./parameters.p', 'rb'))
output_interval = (10/(df_params['dtg']/df_params['tg'])).round().astype(int)[ID]
dt = output_interval*df_params['dtg'][ID]

####### calculate elevation change
z_change = np.zeros((len(files),7))
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

    relief_change[i,0] = np.sum(elev*grid.cell_area_at_node)
    relief_change[i,1] = (relief_change[i,0]- relief_change[i-1,0])/dt

    elev0 = elev.copy()

    # if i%100==0:
    #
    #     field = np.log10((elev_diff/dt)/df_params['U'][ID])
    #     y = np.arange(grid.shape[0] + 1) * grid.dy - grid.dy * 0.5
    #     x = np.arange(grid.shape[1] + 1) * grid.dx - grid.dx * 0.5
    #
    #     plt.figure(figsize=(8,6))
    #     im = plt.imshow(field.reshape(grid.shape).T,
    #                      origin="lower",
    #                      extent=(x[0], x[-1], y[0], y[-1]),
    #                      cmap='plasma',
    #                      vmin=-10,
    #                      vmax=0.0,
    #                      )
    #     plt.colorbar(im, label='log10(dzdt/U)')
    #     plt.title('ID %d, Iteration %d'%(ID,i))
    #     plt.savefig('../post_proc/%s/elev_change_%d_%d.png'%(base_output_path, ID, i))
    #     plt.close()


        # cmap = plt.get_cmap('plasma')
        # fig = plt.figure()
        # sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=-10, vmax=1.0))
        # rgb = sm.to_rgba(field.reshape(grid.shape).T, alpha=0.7)
        # ls = LightSource(azdeg=135, altdeg=45)
        # sr = ls.shade_rgb(rgb, elev.reshape(grid.shape).T, blend_mode='hsv', vert_exag=1.2, dx=grid.dx, dy=grid.dy)
        # plt.imshow(sr, origin="lower", extent=(x[0], x[-1], y[0], y[-1]),)
        # cbar = fig.colorbar(sm, label='log10(dzdt/U)')
        # # cbar.solids.set_edgecolor("face")
        # plt.title('ID %d, Iteration %d'%(ID,i))
        # plt.savefig('../post_proc/%s/elev_change_hs_%d_%d.png'%(base_output_path, ID, i))
        # plt.close()

df_z_change = pd.DataFrame(z_change,columns=['max', '90 perc', '50 perc', '10 perc', 'min', 'mean'])
df_z_change.to_csv('../post_proc/%s/z_change_%d.csv'%(base_output_path, ID))

r_change = pd.DataFrame()
r_change['r_nd'] = relief_change[:,0]/(df_params['hg'][ID]*df_params['lg'][ID]**2)
r_change['drdt_nd'] = relief_change[:,1]/(df_params['hg'][ID]*df_params['lg'][ID]**2)*df_params['tg'][ID]
r_change['t_nd'] = np.arange(len(files))*(dt/df_params['tg'][ID])
# df_z_change.to_csv('../post_proc/%s/z_change_steady_state_%d.csv'%(base_output_path, ID))
r_change.to_csv('../post_proc/%s/relief_change_%d.csv'%(base_output_path, ID))
