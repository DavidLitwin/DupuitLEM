"""
Generate parameters for StreamPowerModel with
-- HydrologyEventVadoseStreamPower
-- FastscapeEroder
-- LinearDiffuser or TaylorNonLinearDiffuser
-- RegolithConstantThickness

Vary parameters around something like the Oregon Coast Range

alpha = hg / lg
gamma = (b ksat hg) / (p lg^2)
beta (formerly Hi) = (ksat hg^2) / (p lg^2)
sigma = (b n) / (p (tr + tb))
rho = tr / (tr + tb)
ai = (pet tb) / (p (tr + tb))
phi = na / ne

17 Nov 2023
"""
#%%
import os
import numpy as np
from itertools import product
import pandas

calc_lg = lambda D, K, v0: (D**2/(v0 * K**2))**(1/3)
calc_tg = lambda D, K, v0: (D/(v0**2 * K**4))**(1/3)
calc_hg = lambda D, U, K, v0: ((D * U**3)/(v0**2 * K**4))**(1/3)
calc_alpha = lambda hg, lg: hg / lg
calc_gamma = lambda b, ksat, hg, lg, p: (b * ksat * hg) / (p * lg**2)
calc_beta = lambda ksat, hg, lg, p: (ksat * hg**2) / (p * lg**2)

#%%

task_id = os.environ['SLURM_ARRAY_TASK_ID']
ID = int(task_id)

#%%
# set up params
K_all = np.geomspace(4e-6,2e-5,5)/(365*24*3600) # s-1
ksat_all = np.geomspace(0.5,2.5,5) /(24*3600) # m/s
prod = np.array(list(product(K_all, ksat_all)))
df_params = pandas.DataFrame(prod, columns=['K', 'ksat'])

df_params['Sc'] = 1.2 # critical slope
df_params['D'] = 0.003/(365*24*3600) # m2/s
df_params['U'] = 1e-4/(365*24*3600) # m/s
df_params['E0'] = 0.0 # don't use threshold model
df_params['v0'] = 20 # m

# estimated poisson parameters for HJAndrews (from script get_event_lengths.py)
tr = 6.34 * 3600 # s
tb = 22.58 * 3600 # s
ds = 6.86 / 1000 # m
df_params['p'] = ds/(tr+tb) # m/s
df_params['RE'] = 0.75 # recharge efficiency (Q/P for Mack Creek)
df_params['ne'] = 0.1
df_params['b'] = 2.0

df_params['BCs'] = 4441
df_params['Nx'] = 300 # number of grid cells width
df_params['Ny'] = 200 # number of grid cells height

df_params['Tg'] = 5e7 * 3600 * 24 * 365 # Total geomorphic simulation time [s]
df_params['Th'] = 10*(tr+tb) # hydrologic simulation time [s]
df_params['dtg'] = 500 * 3600 * 24 * 365 # geomorphic timestep [s]
df_params['ksf'] = df_params['dtg']/df_params['Th'] # morphologic scaling factor
df_params['dtg_max'] = 600 # the maximum duration of a geomorphic substep [s]
df_params['output_interval'] = 2000

df_params['lg'] = calc_lg(df_params.D, df_params.K, df_params.v0)
df_params['hg'] = calc_hg(df_params.D, df_params.U, df_params.K, df_params.v0)
df_params['tg'] = calc_tg(df_params.D, df_params.K, df_params.v0)
df_params['alpha'] = calc_alpha(df_params.hg, df_params.lg)
df_params['gamma'] = calc_gamma(df_params.b, df_params.ksat, df_params.hg, df_params.lg, df_params.p)
df_params['beta'] = calc_beta(df_params.ksat, df_params.hg, df_params.lg, df_params.p)

#%%
df_params.loc[ID].to_csv('parameters.csv', index=True)

