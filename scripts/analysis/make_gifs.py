# -*- coding: utf-8 -*-
"""
Make gif plots of elevation evolution for a given run

Created on Wed Jul  1 15:19:58 2020

@author: dgbli
"""
import matplotlib.pyplot as plt
from landlab import imshow_grid
from landlab.io.netcdf import from_netcdf
import numpy as np
import os
import glob
import imageio

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_output_path = os.environ['BASE_OUTPUT_FOLDER']


grid_files = glob.glob('./data/*.nc')
files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))

indices = []
for i in range(len(files)):
    indices.append(int(files[i].split('_')[-1][:-3]))

path = files[-1]
grid = from_netcdf(path)
elev = grid.at_node['topographic__elevation']


def plot_elev(file,index):

    grid = from_netcdf(file)

    fig = plt.figure(figsize=(8,6))
    imshow_grid(grid,'topographic__elevation', cmap='gist_earth', limits=(0,max(elev)), colorbar_label = 'Elevation [m]', grid_units=('m','m'))
    plt.tight_layout()

    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return image

imageio.mimsave('../post_proc/%s/elev_ID_%d.gif'%(base_output_path, ID), [plot_elev(files[i],indices[i]) for i in range(len(files))], fps=2)
