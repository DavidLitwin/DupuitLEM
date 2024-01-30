"""
Generate parameters for simple streampower model, varying K, D, U, and v0.

New Nov. 2023.

"""

import os
import numpy as np
import pandas as pd
from itertools import product

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)


D_all = [0.005, 0.0005]
U_all = [1e-4, 1e-5]
K_all = np.linspace(1e-4, 1e-5, 10)
v0_all = [5, 10, 50]
Nx = 200
Ny = 400
T = 5e7
dt = 100
m = 0.5
n = 1
Sc = 0.0
# routing_method = 'D8'
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
# df_params['routing_method'] = routing_method
df_params['r_condition'] = r_condition
df_params['output_interval'] = 500
df_params['BCs'] = 4141

df_params.loc[ID].to_csv('parameters.csv', index=True)
