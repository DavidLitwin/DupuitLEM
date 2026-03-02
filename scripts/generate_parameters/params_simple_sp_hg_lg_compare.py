"""
Generate parameters for simple streampower model, varying hg.
Includes generalised characteristic scales (any m, n) from nondimensionalisation.

"""
#%%
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

hg_all = [1.875, 3.75, 7.5]
lg = 15
tg = 1e4
v0 = 30
dt_nd = 0.05
T_nd = 500

T = T_nd * tg
dt = dt_nd * tg
m = 0.5
n = 1.0
Nx = 200
Ny = 200
sc = 1.25 # 0.0
r_condition = 0.0 #1e-8

df_params = pd.DataFrame(hg_all, columns=['hg'])
df_params['v0'] = v0
df_params['lg'] = lg
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
df_params['alpha'] = df_params['hg']/df_params['lg']
df_params['Sc'] = sc

df_params['ksn_pred'] = (df_params['U']/df_params['K'])**(1/df_params['n'])

df_params['r_condition'] = r_condition
df_params['output_interval'] = 500
# df_params['BCs'] = 4141

#%%

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)
df_params.loc[ID].to_csv('parameters.csv', index=True)
