"""
Generate parameters for simple streampower model, varying K, D, U, and v0.
Includes generalised characteristic scales (any m, n) from nondimensionalisation.

New Nov. 2023.

"""

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

D_all = [1e-2, 1e-3]
U_all = [5e-4, 1e-4]
K_all = np.linspace(5e-5, 5e-6, 5)
v0_all = [5, 10, 20]
Nx = 400
Ny = 200
T = 5e7
dt = 250
m = 0.5
n = 1.0
Sc = 0.0
routing_method = 'D8'
r_condition = 0.0 #1e-8

prod = np.array(list(product(K_all, D_all, U_all, v0_all)))
df_params = pd.DataFrame(prod, columns=['K', 'D', 'U', 'v0'])
df_params['Nx'] = Nx
df_params['Ny'] = Ny
df_params['T'] = T
df_params['dt'] = dt
df_params['m'] = m
df_params['n'] = n
df_params['Sc'] = Sc
df_params['routing_method'] = routing_method
df_params['r_condition'] = r_condition
df_params['output_interval'] = 500
df_params['BCs'] = 4141

# generalised characteristic scales
df_params['tg'] = calc_tg(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n, df_params.v0)
df_params['hg'] = calc_hg(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n, df_params.v0)
df_params['lg'] = calc_lg(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n, df_params.v0)

# T+K generalised scales
df_params['lc'] = calc_lc(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n)
df_params['hc'] = calc_hc(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n)
df_params['tc'] = calc_tc(df_params.K, df_params.D, df_params.U, df_params.m, df_params.n)

df_params['v0/lc'] = df_params['v0']/df_params['lc']
df_params['v0/lg'] = df_params['v0']/df_params['lg']


df_params.loc[ID].to_csv('parameters.csv', index=True)
