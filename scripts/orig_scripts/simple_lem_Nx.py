# -*- coding: utf-8 -*-
"""
NoHyd model Q*=1 for all (x,y).

This script uses characteristic scales and dimensionless parameters presented
in Litwin et al. 2021. Vary v0 for one combination of hg and lg.

Date: 2 Jan 2020
"""

import os
import numpy as np
import pandas
from tqdm import tqdm

from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    FastscapeEroder,
    LinearDiffuser,
    )
from landlab.io.netcdf import to_netcdf

#dim equations
def K_fun(v0, lg, tg):
    return np.sqrt(lg)/(np.sqrt(v0)*tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

def generate_parameters(hg, lg, tg, v0, Nx):
    K = K_fun(v0, lg, tg)
    D = D_fun(lg, tg)
    U = U_fun(hg, tg)
    return K, D, U, v0, hg, lg, tg, Nx

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
base_path = './data/simple_lem_'

lg = 15
hg = 2.25
v0 = 0.7*lg # contour width (also grid spacing) [m]
tg = 22500 # geomorphic timescale [yr]

K = K_fun(v0, lg, tg)
D = D_fun(lg, tg)
U = U_fun(hg, tg)
Nx_all = [25, 50, 100, 200, 300]

params = np.zeros((len(Nx_all),8))
for i in range(len(Nx_all)):
    params[i,:] = generate_parameters(hg, lg, tg, v0, Nx_all[i])
df_params = pandas.DataFrame(params,columns=['K', 'D', 'U', 'v0', 'hg', 'lg', 'tg', 'Nx'])
df_params['dt'] = 4e-3*tg
df_params['T'] = 500*tg
df_params['alpha'] = df_params['hg']/df_params['lg']

df_params.to_csv('parameters.csv', index=True)

Ksp = df_params['K'][ID]
v0 = df_params['v0'][ID]
D = df_params['D'][ID]
U = df_params['U'][ID]
hg = df_params['hg'][ID]
lg = df_params['lg'][ID]
tg = df_params['tg'][ID]
dt = df_params['dt'][ID]
T = df_params['T'][ID]
Nx = df_params['Nx'][ID]

N = int(T//dt)
output_interval = 2000

np.random.seed(12345)
grid = RasterModelGrid((Nx, Nx), xy_spacing=v0)
grid.set_status_at_node_on_edges(right=grid.BC_NODE_IS_CLOSED,
                                top=grid.BC_NODE_IS_CLOSED,
                                left=grid.BC_NODE_IS_FIXED_VALUE,
                                bottom=grid.BC_NODE_IS_CLOSED,
)
z = grid.add_zeros('node', 'topographic__elevation')
z[:] = 0.1*hg*np.random.rand(len(z))

fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8', depression_finder='LakeMapperBarnes', method='D8')
ld = LinearDiffuser(grid, linear_diffusivity=D)
sp = FastscapeEroder(grid, K_sp=Ksp, m_sp=0.5, n_sp=1.0)

for i in tqdm(range(N), desc="Completion"):

    z[grid.core_nodes] += U*dt

    ld.run_one_step(dt)
    fa.run_one_step()
    sp.run_one_step(dt)

    # print('completed loop %d'%i)

    if i%output_interval==0:
        print('finished iteration %d'%i)
        filename = base_path + '%d_grid_%d.nc'%(ID,i)
        to_netcdf(grid, filename, include="at_node:topographic__elevation")
