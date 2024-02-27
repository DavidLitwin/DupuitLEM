"""
Generate parameters for simple streampower model, varying K, D, U, and v0.
Includes generalised characteristic scales (any m, n) from nondimensionalisation.

New Nov. 2023.

"""

import os
import numpy as np
import pandas as pd
from itertools import product

#dim equations
def K_fun(hg, lg, tg, v0, m, n):
    return (hg**(1-n) * lg**(n-m)) / (v0**m * tg)

def D_fun(lg, tg):
    return lg**2/tg

def U_fun(hg, tg):
    return hg/tg

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

hg_all = np.linspace(5, 50, 5)
lg_all = np.linspace(10, 50, 5)
tg = 1e4
v0_nd_all = np.array([0.5, 1.0, 2.0])
dt_nd = 0.05
T_nd = 500

T = T_nd * tg
dt = dt_nd * tg
m = 0.5
n = 1.0
Nx = 400
Ny = 200
r_condition = 0.0 #1e-8

prod = np.array(list(product(hg_all, lg_all, v0_nd_all)))
df_params = pd.DataFrame(prod, columns=['hg', 'lg', 'v0_nd'])
df_params['v0'] = df_params['v0_nd'] * df_params['lg']
df_params['tg'] = tg
df_params['K'] = K_fun(df_params['hg'], df_params['lg'], df_params['tg'], df_params['v0'], m, n)
df_params['D'] = D_fun(df_params['lg'], df_params['tg'])
df_params['U'] = U_fun(df_params['hg'], df_params['tg'])
df_params['Nx'] = Nx
df_params['Ny'] = Ny
df_params['T'] = T
df_params['dt'] = dt
df_params['m'] = m
df_params['n'] = n

df_params['ksn_pred'] = (df_params['U']/df_params['K'])**(1/df_params['n'])

df_params['r_condition'] = r_condition
df_params['output_interval'] = 500
df_params['BCs'] = 4141

df_params.loc[ID].to_csv('parameters.csv', index=True)
