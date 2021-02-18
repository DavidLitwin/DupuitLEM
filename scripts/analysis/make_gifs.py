# -*- coding: utf-8 -*-
"""
Make gif plots of elevation evolution for a given run
"""
import numpy as np
import os
import glob
import imageio

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from landlab import imshow_grid
from landlab.io.netcdf import from_netcdf


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


def plot_elev(file, index):
    grid = from_netcdf(file)

    fig = plt.figure(figsize=(8,6))
    imshow_grid(grid,'topographic__elevation', cmap='gist_earth', limits=(0,max(elev)), colorbar_label = 'Elevation [m]', grid_units=('m','m'))
    plt.tight_layout()

    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()
    return image

def plot_hillshade(file, index):
    grid = from_netcdf(file)
    elev_plt = grid.at_node['topographic__elevation']
    y = np.arange(grid.shape[0] + 1) * grid.dy - grid.dy * 0.5
    x = np.arange(grid.shape[1] + 1) * grid.dx - grid.dx * 0.5

    ls = LightSource(azdeg=135, altdeg=45)
    fig = plt.figure(figsize=(8,6))
    plt.imshow(ls.hillshade(elev_plt.reshape(grid.shape).T, vert_exag=2, dx=grid.dx, dy=grid.dy), origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap='gray')
    plt.tight_layout()

    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()
    return image

imageio.mimsave('../post_proc/%s/elev_ID_%d.gif'%(base_output_path, ID), [plot_elev(files[i],indices[i]) for i in range(len(files))], fps=12)

imageio.mimsave('../post_proc/%s/hillshade_ID_%d.gif'%(base_output_path, ID), [plot_hillshade(files[i],indices[i]) for i in range(len(files))], fps=12)
