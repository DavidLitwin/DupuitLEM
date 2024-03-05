"""
Generate parameters for simple streampower model, varying K, D, U, and v0.
Includes generalised characteristic scales (any m, n) from nondimensionalisation.

New Nov. 2023.

"""
#%%
import os
import numpy as np
import pandas as pd
from itertools import product

# generalised characteristic scales from Litwin et al (2022)
def calc_tg(K, D, U, m, n, v0):
    return ((D**(n-m) * U**(2-2*n))/(v0**(2*m) * K**2))**(1/(m+n))
def calc_hg(K, D, U, m, n, v0):
    return ((D**(n-m) * U**(2+m-n))/(v0**(2*m) * K**2))**(1/(m+n))
def calc_lg(K, D, U, m, n, v0):
    return ((D**n * U**(1-n))/(v0**m * K))**(1/(m+n))

# generalised characteristic scales from T+K (2018)
def calc_tc(K, D, U, m, n):
    return (K**(-2) * D**(n-2*m) * U**(2-2*n))**(1/(n+2*m))
def calc_hc(K, D, U, m, n):
    return (K**(-2) * D**(n-2*m) * U**(2-n+2*m))**(1/(n+2*m))
def calc_lc(K, D, U, m, n):
    return (K**(-1) * D**n * U**(1-n))**(1/(n+2*m))

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

D_all = [5e-3, 1e-2]
U_all = [1e-4, 1e-3]
# K_all = np.linspace(2e-5,8e-5, 10)
# m = 0.5
# n = 1.0

K_all = np.linspace(2e-5, 8e-5, 10)
m = 0.8
n = 2.0

# K_all = np.linspace(5e-5, 1e-4, 10)
# m = 0.4
# n = 0.6
v0_all = [10, 20, 50]
Nx = 400
Ny = 200
dt_nd = 0.05
T_nd = 250

routing_method = 'D8'
r_condition = 0.0 #1e-8

prod = np.array(list(product(K_all, D_all, U_all, v0_all)))
df_params = pd.DataFrame(prod, columns=['K', 'D', 'U', 'v0'])
df_params['m'] = m
df_params['n'] = n

# generalised characteristic scales
df_params['tg'] = calc_tg(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n, df_params.v0)
df_params['hg'] = calc_hg(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n, df_params.v0)
df_params['lg'] = calc_lg(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n, df_params.v0)

# T+K generalised scales
df_params['lc'] = calc_lc(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n)
df_params['hc'] = calc_hc(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n)
df_params['tc'] = calc_tc(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n)

df_params['lc/v0'] = df_params['lc']/df_params['v0']
df_params['lg/v0'] = df_params['lg']/df_params['v0']
# print('lg/v0:', min(df_params['lg/v0']),max(df_params['lg/v0']) )

df_params['ksn_pred'] = (df_params['U']/df_params['K'])**(1/df_params['n'])
# print('ksn:', min(df_params['ksn_pred']),max(df_params['ksn_pred']) )

df_params['Nx'] = Nx
df_params['Ny'] = Ny
df_params['T'] = T_nd * df_params['tg']
df_params['dt'] = dt_nd * df_params['tg']
df_params['routing_method'] = routing_method
df_params['r_condition'] = r_condition
df_params['output_interval'] = 500
df_params['BCs'] = 4141

#%%
df_params.loc[ID].to_csv('parameters.csv', index=True)
