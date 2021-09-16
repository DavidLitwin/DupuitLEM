# -*- coding: utf-8 -*-
"""
NoHyd model Q*=1 for all (x,y).

This script uses characteristic scales and dimensionless parameters presented
in Litwin et al. 2021. Vary lg and hg to confirm the independence of these
length scales. Include a streampower erosion threshold.

Date: 2 Jan 2020
"""

import os
import numpy as np
import pandas
import pickle
from itertools import product

from landlab import RasterModelGrid
from landlab.components import (
    LinearDiffuser,
    FlowAccumulator,
    FastscapeEroder,
    LakeMapperBarnes,
    DepressionFinderAndRouter,
    )
from landlab.io.netcdf import to_netcdf

#dim equations
def K_fun(v0, lg, tg):
    return np.sqrt(lg)/(np.sqrt(v0)*tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

def E0_fun(Ntheta, hg, tg):
    return Ntheta*(hg/tg)

def generate_parameters(hg, lg, tg, v0, Ntheta):
    K = K_fun(v0, lg, tg)
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    E0 = E0_fun(Ntheta, hg, tg)
    return K, D, U, v0, hg, lg, tg, E0, Ntheta

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_path = './data/simple_lem_5_'

lg_1 = np.array([15, 30, 60]) # geomorphic length scale [m]
hg_1 = np.array([2.25, 4.5, 9]) # geomorphic height scale [m]
lg_all = np.array(list(product(lg_1, hg_1)))[:,0]
hg_all = np.array(list(product(lg_1, hg_1)))[:,1]
v0_all = 0.7*lg_all # contour width (also grid spacing) [m]
tg = 22500 # geomorphic timescale [yr]
Ntheta = 1.0

params = np.zeros((len(lg_all),9))
for i in range(len(lg_all)):
    params[i,:] = generate_parameters(hg_all[i], lg_all[i], tg, v0_all[i], Ntheta)
df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'v0', 'hg', 'lg', 'tg', 'E0', 'Ntheta'])
df_params['dt'] = 2e-3*tg
df_params['T'] = 500*tg
df_params['alpha'] = df_params['hg']/df_params['lg']

pickle.dump(df_params, open('parameters.p','wb'))

Ksp = df_params['K'][ID]
v0 = df_params['v0'][ID]
E0 = df_params['E0'][ID]
D = df_params['D'][ID]
U = df_params['U'][ID]
hg = df_params['hg'][ID]
lg = df_params['lg'][ID]
tg = df_params['tg'][ID]
dt = df_params['dt'][ID]
T = df_params['T'][ID]

N = int(T//dt)
output_interval = 2000

np.random.seed(12345)
grid = RasterModelGrid((125,125), xy_spacing=v0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED,
                                top=grid.BC_NODE_IS_CLOSED,
                                left=grid.BC_NODE_IS_FIXED_VALUE,
                                bottom=grid.BC_NODE_IS_CLOSED,
)
z = grid.add_zeros('node', 'topographic__elevation')
z[:] = 0.1*hg*np.random.rand(len(z))

fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8')
lmb = LakeMapperBarnes(
    grid,
    method='D8',
    fill_flat=False,
    surface="topographic__elevation",
    fill_surface="topographic__elevation",
    redirect_flow_steepest_descent=False,
    reaccumulate_flow=False,
    track_lakes=False,
    ignore_overfill=True,
)
dfr = DepressionFinderAndRouter(grid)


ld = LinearDiffuser(grid, D)
sp = FastscapeEroder(grid, K_sp=Ksp, m_sp=0.5, n_sp=1.0, threshold_sp=E0)

for i in range(N):
    
    dfr._find_pits()
    if dfr._number_of_pits > 0:
        lmb.run_one_step()
        
    z[grid.core_nodes] += U*dt

    ld.run_one_step(dt)
    fa.run_one_step()
    sp.run_one_step(dt)

    print('completed loop %d'%i)

    if i%output_interval==0:
        print('finished iteration %d'%i)
        filename = base_path + '%d_grid_%d.nc'%(ID,i)
        to_netcdf(grid, filename, include="at_node:topographic__elevation")
